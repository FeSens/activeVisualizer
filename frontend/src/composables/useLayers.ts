import baseApi from "../api/baseApi";
import { ref, Ref } from 'vue';

export default function useLayers() {
  const layers: Ref<{ name: string, username: string }[]> = ref([{ name: '', username: '' }]);

  const getLayers = async () => {
    const response = await baseApi.getCapturableLayers();
    const data = response.data.model_layers;
    //  generate a list of layers {name and shape}
    const layerList = Object.keys(data).map((layer) => ({
      name: layer.replace('.shape', ''),
      username: data[layer].toString()
    }));
    layers.value = layerList;
  };

  return {
    getLayers,
    layers
  };
}