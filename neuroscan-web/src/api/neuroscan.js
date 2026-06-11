import axios from 'axios';

export function createClient(baseURL) {
  return axios.create({ baseURL: baseURL.replace(/\/$/, '') });
}

export async function predict(baseURL, imageFile, model = 'efficientnet_b3', onUploadProgress) {
  const client = createClient(baseURL);
  const form = new FormData();
  form.append('file', imageFile);

  const { data } = await client.post(`/predict?model=${model}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
    timeout: 60_000,
  });

  return data;
}

export async function healthCheck(baseURL) {
  const client = createClient(baseURL);
  const { data } = await client.get('/health', { timeout: 5000 });
  return data;
}
