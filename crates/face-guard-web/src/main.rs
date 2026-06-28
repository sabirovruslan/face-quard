#[cfg(target_arch = "wasm32")]
mod app {
    use gloo_net::http::Request;
    use leptos::ev::{Event, SubmitEvent};
    use leptos::prelude::*;
    use serde::{Deserialize, Serialize};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::spawn_local;
    use web_sys::{FormData, HtmlInputElement, Url};

    const RECENT_KEYS_STORAGE_KEY: &str = "faceGuard.recentKeys";
    const API_URL_STORAGE_KEY: &str = "faceGuard.apiUrl";

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct UploadObjectResponse {
        image_key: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct CreateFaceImageResponse {
        id: String,
        status: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct SearchFaceResponse {
        collection_slug: String,
        matches: Vec<SearchFaceMatchResponse>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct SearchFaceMatchResponse {
        id: String,
        image_key: String,
        similarity: f32,
    }

    #[component]
    pub fn App() -> impl IntoView {
        let api_url = RwSignal::new(
            local_storage_get(API_URL_STORAGE_KEY)
                .unwrap_or_else(|| "http://localhost:8080".to_string()),
        );
        let health = RwSignal::new("Not checked".to_string());
        let health_class = RwSignal::new("health".to_string());

        let collection = RwSignal::new("test_collection".to_string());
        let recent_keys = RwSignal::new(load_recent_keys());

        let selected_file_meta = RwSignal::new("None".to_string());
        let file_preview_url = RwSignal::new(String::new());
        let upload_output = RwSignal::new(String::new());
        let create_output = RwSignal::new(String::new());
        let search_output = RwSignal::new(String::new());
        let matches = RwSignal::new(Vec::<SearchFaceMatchResponse>::new());

        let upload_busy = RwSignal::new(false);
        let create_busy = RwSignal::new(false);
        let search_busy = RwSignal::new(false);

        let check_health = move |event: SubmitEvent| {
            event.prevent_default();
            health.set("Checking...".to_string());
            health_class.set("health".to_string());

            let base_url = api_base(api_url);
            let url = format!("{base_url}/health");

            spawn_local(async move {
                match Request::get(&url).send().await {
                    Ok(response) if response.ok() => {
                        health.set("Backend reachable".to_string());
                        health_class.set("health ok".to_string());
                        local_storage_set(API_URL_STORAGE_KEY, &base_url);
                    }
                    Ok(response) => {
                        health.set(format!("{} {}", response.status(), response.status_text()));
                        health_class.set("health error".to_string());
                    }
                    Err(error) => {
                        health.set(error.to_string());
                        health_class.set("health error".to_string());
                    }
                }
            });
        };

        let on_file_change = move |event: Event| {
            let Some(input) = event
                .target()
                .and_then(|target| target.dyn_into::<HtmlInputElement>().ok())
            else {
                return;
            };

            let Some(file) = input.files().and_then(|files| files.get(0)) else {
                selected_file_meta.set("None".to_string());
                file_preview_url.set(String::new());
                return;
            };

            let file_name = file.name();
            let safe_name = sanitize_file_name(&file_name);
            let current_key = input_value("upload-key");

            if current_key.trim().is_empty() {
                set_input_value(
                    "upload-key",
                    &format!("faces/{}-{safe_name}", js_sys::Date::now() as u64),
                );
            }

            selected_file_meta.set(format!(
                "{} · {} KB · {}",
                file_name,
                (file.size() / 1024.0).ceil() as u64,
                fallback(&file.type_(), "unknown")
            ));

            if let Ok(url) = Url::create_object_url_with_blob(&file) {
                file_preview_url.set(url);
            }
        };

        let submit_upload = move |event: SubmitEvent| {
            event.prevent_default();

            let image_key = input_value("upload-key");
            let Some(file) = input_file("upload-file") else {
                upload_output.set("Select a file".to_string());
                return;
            };

            upload_busy.set(true);
            upload_output.set("Uploading...".to_string());

            let url = format!("{}/api/v1/objects/upload", api_base(api_url));

            spawn_local(async move {
                let result = async {
                    let form = FormData::new().map_err(js_error)?;
                    form.append_with_str("image_key", &image_key)
                        .map_err(js_error)?;
                    form.append_with_blob("file", &file).map_err(js_error)?;

                    let response = Request::post(&url)
                        .body(form)
                        .map_err(|error| error.to_string())?
                        .send()
                        .await
                        .map_err(|error| error.to_string())?;

                    parse_json_response::<UploadObjectResponse>(response).await
                }
                .await;

                match result {
                    Ok(body) => {
                        upload_output.set(pretty_json(&body));
                        set_input_value("create-key", &body.image_key);
                        set_input_value("search-key", &body.image_key);
                        remember_key(recent_keys, body.image_key);
                    }
                    Err(error) => upload_output.set(error),
                }

                upload_busy.set(false);
            });
        };

        let submit_create = move |event: SubmitEvent| {
            event.prevent_default();

            let image_key = input_value("create-key");
            let collection_slug = collection.get_untracked();
            let url = format!("{}/api/v1/faces/create", api_base(api_url));

            create_busy.set(true);
            create_output.set("Creating...".to_string());

            spawn_local(async move {
                let payload = serde_json::json!({
                    "image_key": image_key,
                    "collection_slug": nullable_string(collection_slug),
                });

                match post_json::<CreateFaceImageResponse>(&url, payload).await {
                    Ok(body) => {
                        create_output.set(pretty_json(&body));
                        remember_key(recent_keys, input_value("create-key"));
                    }
                    Err(error) => create_output.set(error),
                }

                create_busy.set(false);
            });
        };

        let submit_search = move |event: SubmitEvent| {
            event.prevent_default();

            let image_key = input_value("search-key");
            let collection_slug = collection.get_untracked();
            let max_faces = input_value("max-faces").parse::<usize>().unwrap_or(10);
            let similarity_threshold = input_value("threshold").parse::<f32>().unwrap_or(80.0);
            let url = format!("{}/api/v1/faces/search_similar", api_base(api_url));

            search_busy.set(true);
            search_output.set("Searching...".to_string());
            matches.set(Vec::new());

            spawn_local(async move {
                let payload = serde_json::json!({
                    "image_key": image_key,
                    "collection_slug": nullable_string(collection_slug),
                    "max_faces": max_faces,
                    "similarity_threshold": similarity_threshold,
                });

                match post_json::<SearchFaceResponse>(&url, payload).await {
                    Ok(body) => {
                        matches.set(body.matches.clone());
                        search_output.set(pretty_json(&body));
                        remember_key(recent_keys, input_value("search-key"));
                    }
                    Err(error) => search_output.set(error),
                }

                search_busy.set(false);
            });
        };

        view! {
            <main class="app-shell">
                <header class="topbar">
                    <div>
                        <h1>"Face Guard Console"</h1>
                        <p>"Upload objects, create face embeddings, and run similarity search."</p>
                    </div>
                    <form class="api-form" on:submit=check_health>
                        <label for="api-url">"API Base URL"</label>
                        <div class="api-row">
                            <input
                                id="api-url"
                                type="url"
                                prop:value=move || api_url.get()
                                on:input=move |event| api_url.set(event_target_value(&event))
                            />
                            <button type="submit">"Check"</button>
                        </div>
                        <span class=move || health_class.get()>{move || health.get()}</span>
                    </form>
                </header>

                <section class="workspace">
                    <aside class="side-panel">
                        <label for="collection">"Collection"</label>
                        <input
                            id="collection"
                            type="text"
                            prop:value=move || collection.get()
                            on:input=move |event| collection.set(event_target_value(&event))
                            autocomplete="off"
                        />

                        <div class="key-history">
                            <div class="section-title">"Recent Keys"</div>
                            <div class="recent-list">
                                <Show
                                    when=move || !recent_keys.get().is_empty()
                                    fallback=|| view! { <p class="muted">"No keys yet"</p> }
                                >
                                    <For
                                        each=move || recent_keys.get()
                                        key=|key| key.clone()
                                        children=move |key| {
                                            let use_key = key.clone();
                                            view! {
                                                <div class="recent-key">
                                                    <span>{key}</span>
                                                    <button
                                                        type="button"
                                                        title="Use key"
                                                        on:click=move |_| {
                                                            set_input_value("create-key", &use_key);
                                                            set_input_value("search-key", &use_key);
                                                        }
                                                    >
                                                        "↗"
                                                    </button>
                                                </div>
                                            }
                                        }
                                    />
                                </Show>
                            </div>
                        </div>
                    </aside>

                    <section class="tool-grid">
                        <article class="panel">
                            <div class="panel-header">
                                <div>
                                    <h2>"1. Upload Object"</h2>
                                    <p>"Send an image to the configured S3/MinIO bucket."</p>
                                </div>
                            </div>
                            <form class="stack" on:submit=submit_upload>
                                <label for="upload-file">"Image File"</label>
                                <input
                                    id="upload-file"
                                    type="file"
                                    accept="image/png,image/jpeg,image/webp"
                                    required
                                    disabled=move || upload_busy.get()
                                    on:change=on_file_change
                                />

                                <label for="upload-key">"Image Key"</label>
                                <input
                                    id="upload-key"
                                    type="text"
                                    placeholder="faces/person-1.jpg"
                                    required
                                    disabled=move || upload_busy.get()
                                />

                                <div class="preview-row">
                                    <img
                                        alt=""
                                        src=move || file_preview_url.get()
                                        class:hidden=move || file_preview_url.get().is_empty()
                                    />
                                    <div>
                                        <div class="muted">"Selected file"</div>
                                        <div class="file-meta">{move || selected_file_meta.get()}</div>
                                    </div>
                                </div>

                                <button class="primary" type="submit" disabled=move || upload_busy.get()>
                                    "Upload"
                                </button>
                            </form>
                            <pre class="output">{move || upload_output.get()}</pre>
                        </article>

                        <article class="panel">
                            <div class="panel-header">
                                <div>
                                    <h2>"2. Create Face Image"</h2>
                                    <p>"Detect a face, generate an embedding, and persist it."</p>
                                </div>
                            </div>
                            <form class="stack" on:submit=submit_create>
                                <label for="create-key">"Image Key"</label>
                                <input
                                    id="create-key"
                                    type="text"
                                    placeholder="faces/person-1.jpg"
                                    required
                                    disabled=move || create_busy.get()
                                />

                                <button class="primary" type="submit" disabled=move || create_busy.get()>
                                    "Create"
                                </button>
                            </form>
                            <pre class="output">{move || create_output.get()}</pre>
                        </article>

                        <article class="panel panel-wide">
                            <div class="panel-header">
                                <div>
                                    <h2>"3. Search Similar"</h2>
                                    <p>"Use an uploaded image key as a query against stored embeddings."</p>
                                </div>
                            </div>
                            <form class="search-form" on:submit=submit_search>
                                <div>
                                    <label for="search-key">"Image Key"</label>
                                    <input
                                        id="search-key"
                                        type="text"
                                        placeholder="search/query.jpg"
                                        required
                                        disabled=move || search_busy.get()
                                    />
                                </div>
                                <div>
                                    <label for="max-faces">"Max Faces"</label>
                                    <input
                                        id="max-faces"
                                        type="number"
                                        min="1"
                                        max="100"
                                        value="10"
                                        disabled=move || search_busy.get()
                                    />
                                </div>
                                <div>
                                    <label for="threshold">"Similarity Threshold"</label>
                                    <input
                                        id="threshold"
                                        type="number"
                                        min="0"
                                        max="100"
                                        step="1"
                                        value="80"
                                        disabled=move || search_busy.get()
                                    />
                                </div>
                                <button class="primary" type="submit" disabled=move || search_busy.get()>
                                    "Search"
                                </button>
                            </form>

                            <div class="results">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>"Image Key"</th>
                                            <th>"ID"</th>
                                            <th>"Similarity"</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <Show
                                            when=move || !matches.get().is_empty()
                                            fallback=|| view! {
                                                <tr>
                                                    <td colspan="3" class="empty">"No matches"</td>
                                                </tr>
                                            }
                                        >
                                            <For
                                                each=move || matches.get()
                                                key=|item| item.id.clone()
                                                children=|item| view! {
                                                    <tr>
                                                        <td>{item.image_key}</td>
                                                        <td>{item.id}</td>
                                                        <td class="score">{format!("{:.2}%", item.similarity * 100.0)}</td>
                                                    </tr>
                                                }
                                            />
                                        </Show>
                                    </tbody>
                                </table>
                            </div>
                            <pre class="output">{move || search_output.get()}</pre>
                        </article>
                    </section>
                </section>
            </main>
        }
    }

    async fn post_json<T>(url: &str, payload: serde_json::Value) -> Result<T, String>
    where
        T: for<'de> Deserialize<'de>,
    {
        let response = Request::post(url)
            .header("Content-Type", "application/json")
            .body(payload.to_string())
            .map_err(|error| error.to_string())?
            .send()
            .await
            .map_err(|error| error.to_string())?;

        parse_json_response(response).await
    }

    async fn parse_json_response<T>(response: gloo_net::http::Response) -> Result<T, String>
    where
        T: for<'de> Deserialize<'de>,
    {
        let status = response.status();
        let status_text = response.status_text();
        let text = response.text().await.map_err(|error| error.to_string())?;

        if !(200..300).contains(&status) {
            return Err(if text.is_empty() {
                format!("{status} {status_text}")
            } else {
                text
            });
        }

        serde_json::from_str(&text).map_err(|error| error.to_string())
    }

    fn pretty_json<T>(value: &T) -> String
    where
        T: serde::Serialize,
    {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string())
    }

    fn input_value(id: &str) -> String {
        input_element(id)
            .map(|input| input.value())
            .unwrap_or_default()
    }

    fn set_input_value(id: &str, value: &str) {
        if let Some(input) = input_element(id) {
            input.set_value(value);
        }
    }

    fn input_file(id: &str) -> Option<web_sys::File> {
        input_element(id)
            .and_then(|input| input.files())
            .and_then(|files| files.get(0))
    }

    fn input_element(id: &str) -> Option<HtmlInputElement> {
        web_sys::window()?
            .document()?
            .get_element_by_id(id)?
            .dyn_into::<HtmlInputElement>()
            .ok()
    }

    fn trim_trailing_slash(value: &str) -> String {
        value.trim_end_matches('/').to_string()
    }

    fn api_base(api_url: RwSignal<String>) -> String {
        trim_trailing_slash(&api_url.get_untracked())
    }

    fn remember_key(recent_keys: RwSignal<Vec<String>>, key: String) {
        if key.trim().is_empty() {
            return;
        }

        let mut keys = recent_keys.get_untracked();
        keys.retain(|item| item != &key);
        keys.insert(0, key);
        keys.truncate(10);
        save_recent_keys(&keys);
        recent_keys.set(keys);
    }

    fn nullable_string(value: String) -> serde_json::Value {
        let value = value.trim().to_string();
        if value.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(value)
        }
    }

    fn sanitize_file_name(value: &str) -> String {
        value
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                    character
                } else {
                    '-'
                }
            })
            .collect()
    }

    fn fallback<'a>(value: &'a str, default: &'a str) -> &'a str {
        if value.is_empty() { default } else { value }
    }

    fn local_storage_get(key: &str) -> Option<String> {
        web_sys::window()?
            .local_storage()
            .ok()??
            .get_item(key)
            .ok()?
    }

    fn local_storage_set(key: &str, value: &str) {
        if let Some(storage) =
            web_sys::window().and_then(|window| window.local_storage().ok().flatten())
        {
            let _ = storage.set_item(key, value);
        }
    }

    fn load_recent_keys() -> Vec<String> {
        local_storage_get(RECENT_KEYS_STORAGE_KEY)
            .and_then(|value| serde_json::from_str(&value).ok())
            .unwrap_or_default()
    }

    fn save_recent_keys(keys: &[String]) {
        if let Ok(value) = serde_json::to_string(keys) {
            local_storage_set(RECENT_KEYS_STORAGE_KEY, &value);
        }
    }

    fn js_error(error: wasm_bindgen::JsValue) -> String {
        error
            .as_string()
            .unwrap_or_else(|| "JavaScript error".to_string())
    }

    pub fn run() {
        leptos::mount::mount_to_body(App);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    app::run();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {}
