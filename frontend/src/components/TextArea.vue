<template>
  <div class="text-area-container">
    <!-- Content-editable div for text input and styled display -->
    <div
      contenteditable="true"
      class="editable-text-area w-full h-32 px-3 py-2 text-base border rounded-lg focus:ring focus:ring-indigo-300"
      @input="handleInput"
      v-html="styledText"
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
    const lastKnownCursorPos = ref(0);

    const updateCursorPos = (activeElement) => {
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        let selectedNode = range.startContainer;
        let offset = range.startOffset;

        // Traverse up to a direct child of the contenteditable element
        while (selectedNode && selectedNode.parentNode !== activeElement) {
          const parent = selectedNode.parentNode;
          for (let i = 0; i < parent.childNodes.length; i++) {
            if (parent.childNodes[i] === selectedNode) {
              break;
            }
            offset += parent.childNodes[i].textContent.length;
          }
          selectedNode = parent; // Move up in the DOM tree
        }

        // Now selectedNode should be a direct child of the contenteditable element
        // Calculate the cursor's position within the entire contenteditable
        let charCount = 0;
        for (const node of Array.from(activeElement.childNodes)) {
          if (node === selectedNode) {
            charCount += offset;
            break;
          } else {
            charCount += node.textContent.length;
          }
        }

        lastKnownCursorPos.value = charCount;
      }
    };

    const handleInput = (event) => {
      const activeElement = event.target;
      updateCursorPos(activeElement); // Update cursor position before re-render
      const rawText = event.target.innerText;
      userText.value = rawText; // Update the user's text
      if (isOpen) {
        send(rawText);
      } else {
        console.log('Socket is not open');
      }
    };

    // watch(() => lastKnownCursorPos.value, (newValue) => {
    //   console.log('Cursor position changed:', newValue);
    // });

    watch(() => JSON.parse(data.value), (newValue) => {
      if (newValue && newValue.tokens && newValue.offsets) {
        // ... your token processing logic here
        // Generate the styled text and update styledText.value
        let newText = userText.value;
        newValue.offsets.slice().reverse().forEach((offset, index) => {
          const token = newValue.tokens[index];
          const textSegment = userText.value.substring(offset[0], offset[1]);
          const styledToken = `<span class="token" data-token-id="${token}">${textSegment}</span>`;
          newText = newText.substring(0, offset[0]) + styledToken + newText.substring(offset[1]);
        });

        styledText.value = newText;

        nextTick(() => {
          const activeElement = document.activeElement; // Now we define activeElement
          if (activeElement && activeElement.className.includes("editable-text-area")) {
            // Restore cursor position based on lastKnownCursorPos
            restoreCursorPosition(activeElement, lastKnownCursorPos.value);
          }
        });
      }
    });


    const restoreCursorPosition = (activeElement, cursorPos) => {
      // Restore the cursor position
      const children = Array.from(activeElement.childNodes);
      let charCount = 0;
      let found = false;
      for (const child of children) {
        let nodeLength = child.nodeType === Node.TEXT_NODE ? child.length : child.textContent.length;
        if (charCount + nodeLength > cursorPos) {
          const range = document.createRange();
          range.setStart(child.nodeType === Node.TEXT_NODE ? child : child.firstChild, cursorPos - charCount);
          range.collapse(true);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          found = true;
          break;
        }
        charCount += nodeLength;
      }
      if (!found) {
        // Fallback if the cursor position is at the end
        const range = document.createRange();
        range.selectNodeContents(activeElement);
        range.collapse(false); // Collapse to end
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
    };

    return {
      styledText,
      handleInput,
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
  margin-right: 2px;
  padding: 2px 2px;
  border-radius: 4px;
  background-color: #eee; /* Example background */
}
</style>
