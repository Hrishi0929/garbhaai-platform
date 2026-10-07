// Server-side forwarder: the browser talks to this app's /api/* routes, and
// these routes forward to the FastAPI services. Keeps the internal service URLs
// out of the browser and avoids cross-origin (CORS) setup on every service.
export const INGESTION_URL = process.env.INGESTION_URL ?? "http://localhost:8001";
export const QUALITY_CHECK_URL = process.env.QUALITY_CHECK_URL ?? "http://localhost:8002";
export const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://localhost:8005";

export async function forward(url: string, req: Request): Promise<Response> {
  try {
    const body = await req.formData();
    const upstream = await fetch(url, { method: "POST", body });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json({ detail: "Upstream service unreachable" }, { status: 502 });
  }
}
