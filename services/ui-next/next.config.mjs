/** @type {import('next').NextConfig} */
const nextConfig = {
  // Self-contained server bundle so the Docker image stays small.
  output: "standalone",
};
export default nextConfig;
