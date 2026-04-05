//! Transparent reverse-proxy for remote backend mode.
//!
//! When enabled, the local server forwards all `/api/v1/*` HTTP requests to
//! the remote Koharu instance (running on Colab with a GPU) and streams the
//! response back to the GUI.  The GUI has zero changes — it doesn't know
//! whether the processing happened locally or remotely.

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, Response, StatusCode, header},
    response::IntoResponse,
};

/// Maximum request body size (1 GiB — same as the local API limit).
const MAX_BODY_SIZE: usize = 1024 * 1024 * 1024;

#[derive(Clone)]
struct ProxyState {
    client: reqwest::Client,
    remote_url: String,
}

/// Build an axum [`Router`] that forwards every request to `remote_url`.
///
/// The returned router has its state resolved (`.with_state()` already called),
/// so it can be nested directly in the main server router.
pub fn router(remote_url: String) -> Router {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("failed to create reqwest client for remote proxy");

    let state = ProxyState {
        client,
        remote_url: remote_url.trim_end_matches('/').to_string(),
    };

    Router::new()
        .fallback(handle_proxy)
        .with_state(state)
}

/// The actual proxy handler — reads the incoming request, replays it against
/// the remote backend, and returns the remote response verbatim.
async fn handle_proxy(
    State(state): State<ProxyState>,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let method = req.method().clone();

    // Preserve path + query string.
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());

    let url = format!("{}{}", state.remote_url, path_and_query);

    // --- Forward headers (skip `host` so the remote sees its own host). ------
    let mut forward_headers = reqwest::header::HeaderMap::new();
    for (name, value) in req.headers() {
        if name == header::HOST {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()),
            reqwest::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            forward_headers.insert(n, v);
        }
    }

    // --- Read request body ---------------------------------------------------
    let body_bytes = axum::body::to_bytes(req.into_body(), MAX_BODY_SIZE)
        .await
        .map_err(|e| ProxyError::bad_request(format!("failed to read request body: {e}")))?;

    // --- Build remote request ------------------------------------------------
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::bad_request(format!("invalid method: {e}")))?;

    let mut remote_req = state
        .client
        .request(reqwest_method, &url)
        .headers(forward_headers);

    if !body_bytes.is_empty() {
        remote_req = remote_req.body(body_bytes);
    }

    // --- Send ----------------------------------------------------------------
    let remote_resp = remote_req.send().await.map_err(|e| {
        tracing::error!(url = %url, "remote proxy error: {e}");
        ProxyError::bad_gateway(format!("remote backend unreachable: {e}"))
    })?;

    // --- Stream remote response back to GUI ----------------------------------
    let status = remote_resp.status();
    let resp_headers = remote_resp.headers().clone();
    let resp_body = remote_resp.bytes().await.map_err(|e| {
        ProxyError::bad_gateway(format!("failed to read remote response body: {e}"))
    })?;

    let mut builder = Response::builder().status(status.as_u16());
    for (name, value) in &resp_headers {
        // Re-encode into axum header types.
        if let (Ok(n), Ok(v)) = (
            axum::http::header::HeaderName::from_bytes(name.as_str().as_bytes()),
            axum::http::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(n, v);
        }
    }

    builder
        .body(Body::from(resp_body))
        .map_err(|e| ProxyError::internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

struct ProxyError {
    status: StatusCode,
    message: String,
}

impl ProxyError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    fn bad_gateway(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: msg.into(),
        }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response<Body> {
        (self.status, self.message).into_response()
    }
}
