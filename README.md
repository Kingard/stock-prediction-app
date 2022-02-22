## Build and Deploy with Google Cloud Run
#### Submit the build
`gcloud builds submit --tag gcr.io/first-ml-project-342113/price-prediction-app  --project=first-ml-project-342113`

#### Deploy
`gcloud run deploy --image gcr.io/first-ml-project-342113/price-prediction-app --platform managed  --project=first-ml-project-342113 --allow-unauthenticated`