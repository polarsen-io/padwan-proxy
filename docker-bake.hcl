group "default" {
  targets = ["slim", "laya"]
}

# CI overrides these with docker/metadata-action tags and labels
target "meta-slim" {
  tags = ["padwan-proxy:latest"]
}

target "meta-laya" {
  tags = ["padwan-proxy:laya"]
}

target "slim" {
  inherits = ["meta-slim"]
  context  = "."
}

# Local Laya approval model: CUDA torch, several GB
target "laya" {
  inherits = ["meta-laya"]
  context  = "."
  args     = { LAYA = "1" }
}
