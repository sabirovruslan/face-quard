#[cfg(target_arch = "wasm32")]
mod app {
    use gloo_net::http::Request;
    use leptos::ev::{Event, SubmitEvent};
    use leptos::prelude::*;
    use serde::{Deserialize, Serialize};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::spawn_local;
    use web_sys::{FormData, HtmlInputElement, Url};

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
        download_url: String,
        similarity: f32,
    }

    #[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
    struct ListFaceImagesCursorResponse {
        created_at: String,
        id: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct FaceImageResponse {
        id: String,
        image_key: String,
        download_url: String,
        collection_slug: String,
        status: String,
        created_at: String,
        updated_at: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct ListFaceImagesResponse {
        items: Vec<FaceImageResponse>,
        next_cursor: Option<ListFaceImagesCursorResponse>,
        has_more: bool,
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
        let list_status = RwSignal::new("processed".to_string());
        let list_limit = RwSignal::new(20usize);

        let selected_file_meta = RwSignal::new("None".to_string());
        let file_preview_url = RwSignal::new(String::new());
        let upload_output = RwSignal::new(String::new());
        let create_output = RwSignal::new(String::new());
        let search_output = RwSignal::new(String::new());
        let matches = RwSignal::new(Vec::<SearchFaceMatchResponse>::new());
        let face_images = RwSignal::new(Vec::<FaceImageResponse>::new());
        let face_images_output = RwSignal::new(String::new());
        let current_face_images_cursor = RwSignal::new(None::<ListFaceImagesCursorResponse>);
        let next_face_images_cursor = RwSignal::new(None::<ListFaceImagesCursorResponse>);
        let face_images_cursor_stack =
            RwSignal::new(Vec::<Option<ListFaceImagesCursorResponse>>::new());

        let upload_busy = RwSignal::new(false);
        let create_busy = RwSignal::new(false);
        let search_busy = RwSignal::new(false);
        let face_images_busy = RwSignal::new(false);

        Effect::new(move |_| {
            load_face_images(
                api_url,
                collection,
                list_status,
                list_limit,
                face_images,
                face_images_output,
                face_images_busy,
                current_face_images_cursor,
                next_face_images_cursor,
                None,
            );
        });

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
                    }
                    Err(error) => upload_output.set(error),
                }

                upload_busy.set(false);
            });
        };

        let submit_create = move |event: SubmitEvent| {
            event.prevent_default();

            let image_key = input_value("create-key");
            let created_image_key = image_key.clone();
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
                        set_input_value("search-key", &created_image_key);
                        face_images_cursor_stack.set(Vec::new());
                        load_face_images(
                            api_url,
                            collection,
                            list_status,
                            list_limit,
                            face_images,
                            face_images_output,
                            face_images_busy,
                            current_face_images_cursor,
                            next_face_images_cursor,
                            None,
                        );
                    }
                    Err(error) => create_output.set(error),
                }

                create_busy.set(false);
            });
        };

        let submit_search = move |event: SubmitEvent| {
            event.prevent_default();
            run_search(
                api_url,
                collection,
                search_output,
                matches,
                search_busy,
                input_value("search-key"),
            );
        };

        let submit_face_images = move |event: SubmitEvent| {
            event.prevent_default();
            face_images_cursor_stack.set(Vec::new());
            load_face_images(
                api_url,
                collection,
                list_status,
                list_limit,
                face_images,
                face_images_output,
                face_images_busy,
                current_face_images_cursor,
                next_face_images_cursor,
                None,
            );
        };

        let next_face_images_page = move |_| {
            let Some(cursor) = next_face_images_cursor.get_untracked() else {
                return;
            };

            let current_cursor = current_face_images_cursor.get_untracked();
            face_images_cursor_stack.update(|stack| stack.push(current_cursor));

            load_face_images(
                api_url,
                collection,
                list_status,
                list_limit,
                face_images,
                face_images_output,
                face_images_busy,
                current_face_images_cursor,
                next_face_images_cursor,
                Some(cursor),
            );
        };

        let previous_face_images_page = move |_| {
            let previous_cursor = face_images_cursor_stack
                .with_untracked(|stack| stack.last().cloned())
                .flatten();
            face_images_cursor_stack.update(|stack| {
                stack.pop();
            });

            load_face_images(
                api_url,
                collection,
                list_status,
                list_limit,
                face_images,
                face_images_output,
                face_images_busy,
                current_face_images_cursor,
                next_face_images_cursor,
                previous_cursor,
            );
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
                                            <th>"Image"</th>
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
                                                    <td colspan="4" class="empty">"No matches"</td>
                                                </tr>
                                            }
                                        >
                                            <For
                                                each=move || matches.get()
                                                key=|item| item.id.clone()
                                                children=move |item| {
                                                    let image_key = item.image_key.clone();
                                                    let download_url = item.download_url.clone();
                                                    let download_url_for_visibility = download_url.clone();
                                                    let id = item.id;
                                                    let similarity = item.similarity;
                                                    view! {
                                                        <tr>
                                                            <td>
                                                                <img
                                                                    class="thumb"
                                                                    alt=""
                                                                    src=download_url
                                                                    class:hidden=move || download_url_for_visibility.is_empty()
                                                                />
                                                            </td>
                                                            <td class="key-cell">{image_key}</td>
                                                            <td>{id}</td>
                                                            <td class="score">{format!("{:.2}%", similarity * 100.0)}</td>
                                                        </tr>
                                                    }
                                                }
                                            />
                                        </Show>
                                    </tbody>
                                </table>
                            </div>
                            <pre class="output">{move || search_output.get()}</pre>
                        </article>

                        <article class="panel panel-wide">
                            <div class="panel-header">
                                <div>
                                    <h2>"4. Face Images"</h2>
                                    <p>"Review stored faces and use any row as a search query."</p>
                                </div>
                            </div>
                            <form class="list-form" on:submit=submit_face_images>
                                <div>
                                    <label for="list-status">"Status"</label>
                                    <input
                                        id="list-status"
                                        type="text"
                                        placeholder="processed"
                                        prop:value=move || list_status.get()
                                        on:input=move |event| list_status.set(event_target_value(&event))
                                        disabled=move || face_images_busy.get()
                                    />
                                </div>
                                <div>
                                    <label for="list-limit">"Limit"</label>
                                    <input
                                        id="list-limit"
                                        type="number"
                                        min="1"
                                        max="100"
                                        prop:value=move || list_limit.get().to_string()
                                        on:input=move |event| {
                                            if let Ok(value) = event_target_value(&event).parse::<usize>() {
                                                list_limit.set(value);
                                            }
                                        }
                                        disabled=move || face_images_busy.get()
                                    />
                                </div>
                                <button class="primary" type="submit" disabled=move || face_images_busy.get()>
                                    "Load"
                                </button>
                            </form>

                            <div class="results">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>"Image"</th>
                                            <th>"Image Key"</th>
                                            <th>"Status"</th>
                                            <th>"Created"</th>
                                            <th>"Actions"</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <Show
                                            when=move || !face_images.get().is_empty()
                                            fallback=|| view! {
                                                <tr>
                                                    <td colspan="5" class="empty">"No face images"</td>
                                                </tr>
                                            }
                                        >
                                            <For
                                                each=move || face_images.get()
                                                key=|item| item.id.clone()
                                                children=move |item| {
                                                    let search_key = item.image_key.clone();
                                                    view! {
                                                        <tr>
                                                            <td>
                                                                <img class="thumb" alt="" src=item.download_url.clone() />
                                                            </td>
                                                            <td class="key-cell">{item.image_key}</td>
                                                            <td>{item.status}</td>
                                                            <td>{item.created_at}</td>
                                                            <td>
                                                                <button
                                                                    type="button"
                                                                    disabled=move || search_busy.get()
                                                                    on:click=move |_| {
                                                                        let image_key = search_key.clone();
                                                                        set_input_value("search-key", &image_key);
                                                                        run_search(
                                                                            api_url,
                                                                            collection,
                                                                            search_output,
                                                                            matches,
                                                                            search_busy,
                                                                            image_key,
                                                                        );
                                                                    }
                                                                >
                                                                    "Search"
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    }
                                                }
                                            />
                                        </Show>
                                    </tbody>
                                </table>
                            </div>

                            <div class="pager">
                                <button
                                    type="button"
                                    disabled=move || face_images_busy.get() || face_images_cursor_stack.get().is_empty()
                                    on:click=previous_face_images_page
                                >
                                    "Previous"
                                </button>
                                <button
                                    type="button"
                                    disabled=move || face_images_busy.get() || next_face_images_cursor.get().is_none()
                                    on:click=next_face_images_page
                                >
                                    "Next"
                                </button>
                                <span class="muted">
                                    {move || if next_face_images_cursor.get().is_some() { "More results available" } else { "End of list" }}
                                </span>
                            </div>
                            <pre class="output">{move || face_images_output.get()}</pre>
                        </article>
                    </section>
                </section>
            </main>
        }
    }

    fn load_face_images(
        api_url: RwSignal<String>,
        collection: RwSignal<String>,
        list_status: RwSignal<String>,
        list_limit: RwSignal<usize>,
        face_images: RwSignal<Vec<FaceImageResponse>>,
        face_images_output: RwSignal<String>,
        face_images_busy: RwSignal<bool>,
        current_cursor: RwSignal<Option<ListFaceImagesCursorResponse>>,
        next_cursor: RwSignal<Option<ListFaceImagesCursorResponse>>,
        cursor: Option<ListFaceImagesCursorResponse>,
    ) {
        let url = format!("{}/api/v1/faces/list", api_base(api_url));
        let collection_slug = collection.get_untracked();
        let status = list_status.get_untracked();
        let limit = list_limit.get_untracked();

        face_images_busy.set(true);
        face_images_output.set("Loading...".to_string());

        spawn_local(async move {
            let payload = serde_json::json!({
                "collection_slug": nullable_string(collection_slug),
                "status": nullable_string(status),
                "limit": limit,
                "cursor": cursor.clone(),
            });

            match post_json::<ListFaceImagesResponse>(&url, payload).await {
                Ok(body) => {
                    face_images.set(body.items.clone());
                    current_cursor.set(cursor);
                    next_cursor.set(body.next_cursor.clone());
                    face_images_output.set(pretty_json(&body));
                }
                Err(error) => face_images_output.set(error),
            }

            face_images_busy.set(false);
        });
    }

    fn run_search(
        api_url: RwSignal<String>,
        collection: RwSignal<String>,
        search_output: RwSignal<String>,
        matches: RwSignal<Vec<SearchFaceMatchResponse>>,
        search_busy: RwSignal<bool>,
        image_key: String,
    ) {
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
                }
                Err(error) => search_output.set(error),
            }

            search_busy.set(false);
        });
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
