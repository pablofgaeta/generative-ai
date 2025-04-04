# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

# Agent server service account.
module "concierge_server_service_account" {
  source     = "terraform-google-modules/service-accounts/google"
  version    = "~> 4.5"
  project_id = google_project.demo_project.project_id
  names      = [local.concierge_server_service_account_name]
  project_roles = [
    "${google_project.demo_project.project_id}=>roles/secretmanager.secretAccessor",
    "${google_project.demo_project.project_id}=>roles/bigquery.user",
    "${google_project.demo_project.project_id}=>roles/bigquery.connectionUser",
    "${google_project.demo_project.project_id}=>roles/aiplatform.user",
  ]
  depends_on = [google_project.demo_project]
}

# Frontend demo service account.
module "concierge_app_engine_service_account" {
  source     = "terraform-google-modules/service-accounts/google"
  version    = "~> 4.5"
  project_id = google_project.demo_project.project_id
  names      = [local.concierge_app_engine_service_account_name]
  project_roles = [
    "${google_project.demo_project.project_id}=>roles/cloudbuild.builds.builder",
    "${google_project.demo_project.project_id}=>roles/iam.serviceAccountTokenCreator",
  ]
  depends_on = [google_project.demo_project]
}

# Cloud Build service account.
module "concierge_build_service_account" {
  source     = "terraform-google-modules/service-accounts/google"
  version    = "~> 4.5"
  project_id = google_project.demo_project.project_id
  names      = [local.concierge_build_service_account_name]
  project_roles = [
    "${google_project.demo_project.project_id}=>roles/cloudbuild.builds.builder",
  ]
  depends_on = [google_project.demo_project]
}
