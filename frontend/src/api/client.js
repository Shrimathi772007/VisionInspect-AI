const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

let authToken = null;
let unauthorizedHandler = () => {};

export function setAuthToken(token) {
  authToken = token;
}

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler;
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) {
      return body.detail.map((item) => item.msg).join(", ");
    }
  } catch {
    // response had no JSON body
  }
  return `Request failed with status ${response.status}`;
}

/**
 * @param {string} path
 * @param {{ method?: string, json?: object, formData?: FormData, responseType?: 'json'|'blob'|'none' }} options
 */
export async function apiFetch(path, options = {}) {
  const { method = "GET", json, formData, responseType = "json" } = options;

  const headers = {};
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  let body;
  if (formData) {
    body = formData;
  } else if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { method, headers, body });
  } catch {
    throw new ApiError(0, "Unable to reach the server. Check your connection and try again.");
  }

  if (response.status === 401) {
    unauthorizedHandler();
    throw new ApiError(401, await parseErrorDetail(response));
  }

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }

  if (responseType === "blob") return response.blob();
  if (responseType === "none") return null;
  if (response.status === 204) return null;
  return response.json();
}
