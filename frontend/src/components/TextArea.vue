<template>
  <div class="text-area-container relative">
    <!-- Invisible text area to capture user input -->
    <textarea
      id="hiddenInput"
      class="absolute top-0 left-0 w-full h-32 opacity-0 z-0"
      @input="formatText($event.target.value)"
    ></textarea>

    <!-- Visible div to show the styled text -->
    <div
      class="styled-text-area w-full h-32 px-3 py-2 text-base border rounded-lg"
      v-html="styledText"
    ></div>
  </div>
</template>

<script>
import useWebSockets from '../composables/baseSockets';
import { ref, watch } from 'vue';
export default {
  name: 'TextArea',

  setup() {
    const { tokenizeText } = useWebSockets();
    const { isOpen, data, send } = tokenizeText;
    const styledText = ref(""); // Use a ref to hold the styled HTML content
    const userInput = ref(""); // Use a ref to hold the user's raw input

    function formatText(rawText) {
      userInput.value = rawText; // Update the raw input
      if (isOpen) {
        send(rawText);
        console.log(rawText);
      } else {
        console.log('Socket is not open');
      }
    }

    watch(() => JSON.parse(data.value), (newValue) => {
      console.log(newValue, newValue.tokens);
      window.xxx = newValue;
      if (newValue && newValue.tokens && newValue.offsets) {
        styledText.value = newValue.tokens.map((token, index) => {
          const textSegment = userInput.value.substring(newValue.offsets[index][0], newValue.offsets[index][1]);
          return `<span class="token" data-token-id="${token}">${textSegment}</span>`;
        }).join('');
      }
    });

    return {
      styledText,
      formatText
    }
  },
}
</script>

<style>
.token {
  /* Your token styling goes here */
  display: inline-block;
  margin-right: 5px;
  padding: 2px 5px;
  border-radius: 4px;
  background-color: #eee; /* Example background */
}
</style>
