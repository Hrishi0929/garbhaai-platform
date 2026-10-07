import { QUALITY_CHECK_URL, forward } from "@/lib/proxy";

export async function POST(req: Request) {
  return forward(`${QUALITY_CHECK_URL}/check-quality`, req);
}
