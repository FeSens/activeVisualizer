/* eslint-disable @typescript-eslint/explicit-module-boundary-types */
import axios, { AxiosRequestConfig } from 'axios';

const config: AxiosRequestConfig = {
  responseType: 'json',
  withCredentials: false,
  baseURL: 'http://localhost:8000',
};
const $axios = axios.create(config);

export default {
  getCapturableLayers: async () => (
    $axios.get<{ model_layers: { [layer: string]: string } }>('v1/model_capturable_layers')
  )
};