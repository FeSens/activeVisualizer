<template>
  <div class="text-area-container">
    <!-- Content-editable div for text input and styled display -->
    <div
      contenteditable="true"
      class="editable-text-area w-full h-32 px-3 py-2 text-base border rounded-lg focus:ring focus:ring-indigo-300"
      @input="formatText($event.target.innerText)"
      v-html="styledText"
      @keyup="moveCursorToEnd($event)"
    ></div>
  </div>
</template>

<script>
import useWebSockets from '../composables/baseSockets';
import { ref, watch, nextTick } from 'vue';
export default {
  name: 'TextArea',

  setup() {
    const { tokenizeText } = useWebSockets();
    const { isOpen, data, send } = tokenizeText;
    const styledText = ref(""); // Use a ref to hold the styled HTML content
    const userText = ref(""); // Use a ref to hold the user's input

    function formatText(rawText) {
      userText.value = rawText; // Update the user's text
      if (isOpen) {
        send(rawText);
      } else {
        console.log('Socket is not open');
      }
    }

    watch(() => JSON.parse(data.value), (newValue) => {
      console.log(newValue);
      if (newValue && newValue.tokens && newValue.offsets) {
        // Generate the styled HTML without disrupting the cursor position
        const selection = window.getSelection();
        let cursorPos = selection.anchorOffset;
        const activeElement = document.activeElement;

        // Insert tokens into the user's text
        let newText = userText.value;
        newValue.offsets.slice().reverse().forEach((offset, index) => {
          const token = newValue.tokens[index];
          const textSegment = userText.value.substring(offset[0], offset[1]);
          const styledToken = `<span class="token" data-token-id="${token}">${textSegment}</span>`;
          newText = newText.substring(0, offset[0]) + styledToken + newText.substring(offset[1]);
        });

        styledText.value = newText;

        // Restore the cursor position
        nextTick(() => {
          if (activeElement.className === "editable-text-area") {
            const range = document.createRange();
            const textNode = activeElement.childNodes[0];
            range.setStart(textNode, cursorPos);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
          }
        });
      }
    });

    function moveCursorToEnd(event) {
      if (event.key === "Enter") {
        // Ensure the cursor moves to the end on Enter key
        const editableDiv = event.target;
        nextTick(() => {
          const range = document.createRange();
          const sel = window.getSelection();
          range.selectNodeContents(editableDiv);
          range.collapse(false);
          sel.removeAllRanges();
          sel.addRange(range);
        });
      }
    }

    return {
      styledText,
      formatText,
      moveCursorToEnd
    }
  },
}
</script>

<style>
.editable-text-area {
  /* Apply Tailwind utilities as needed */
  @apply outline-none;
}

.token {
  /* Your token styling goes here */
  display: inline-block;
  margin-right: 5px;
  padding: 2px 5px;
  border-radius: 4px;
  background-color: #eee; /* Example background */
}
</style>
