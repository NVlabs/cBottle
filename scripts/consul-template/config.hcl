consul {
  address = "localhost:8500"
}

template {
  source = "./scripts/nginx/xarray.conf.template"
  destination = "./scripts/nginx/xarray.conf"
  perms       = 0644
  command     = "./restart_nginx.sh"
} 