<template>
  <div class="text-area-container w-full h-32 min-h-32 relative">
    <div
      contenteditable="true"
      class="editable-text-area w-full h-full px-3 py-2 text-base border rounded-lg focus:ring focus:ring-indigo-300 overflow-hidden"
      @input="$emit('textChanged', $event.target.innerText)"
    ></div>
    <div class="absolute bottom-2 right-2 px-2 rounded-full bg-gray-200 text-gray-600">
      {{ tokenCount }}
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
export default {
  name: 'TextArea',
  emit: ['textChanged'],
  props: {
    tokenCount: {
      type: Number,
    },
  },
}

</script>
