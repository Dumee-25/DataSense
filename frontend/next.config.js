/** @type {import("next").NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://localhost:8000/api/:path*" }]
  },
  // Allow large file uploads through the proxy (default is very low)
  experimental: {
    serverActions: {
      bodySizeLimit: "60mb",
    },
  },
}
module.exports = nextConfig
