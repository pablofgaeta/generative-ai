# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

locals {
  concierge_server_service_account_name     = "concierge-server"
  concierge_build_service_account_name      = "concierge-build"
  concierge_app_engine_service_account_name = "concierge-app-engine"
}

resource "google_project" "demo_project" {
  name = var.project_name
  project_id = var.project_id
  billing_account = var.billing_account
  auto_create_network = false
  
}

resource "google_project_service" "services" {
  for_each = toset([
    "serviceusage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "vpcaccess.googleapis.com",
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "servicenetworking.googleapis.com",
    "iam.googleapis.com",
    "alloydb.googleapis.com",
    "appengine.googleapis.com",
    "appengineflex.googleapis.com",
    "iap.googleapis.com",
    "bigquery.googleapis.com",
    "bigqueryconnection.googleapis.com",
  ])

  project = google_project.demo_project.project_id
  service = each.key
  disable_dependent_services = false
  disable_on_destroy = false
}

# Create an Artifact Registry repository for hosting docker images for the concierge demo.
resource "google_artifact_registry_repository" "concierge-repo" {
  project       = google_project.demo_project.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo
  format        = "DOCKER"

  depends_on = [google_project.demo_project]
}
