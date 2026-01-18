"""
Custom CSS styles for Gradio interface.
"""

GRADIO_CUSTOM_CSS = """
        /* Center-align the main heading */
        .gradio-container h1 {
            text-align: center !important;
        }
        /* Set font for entire interface */
        .gradio-container, .gradio-container * {
            font-family: 'Consolas', monospace !important;
        }
        .gradio-container .chatbot {
            height: 70vh !important;
            min-height: 400px;
        }
        /* Align elements by height - works for rows with equal_height (excluding unequal-height) */
        .gradio-container .row:not(.unequal-height) > .column {
            display: flex !important;
            align-items: stretch !important;
        }
        .gradio-container .row:not(.unequal-height) > .column > .block {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
        }
        /* Align text field by height - occupies entire available area (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child .form {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        /* Textarea container occupies full height (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block > label {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block > label > .input-container {
            flex: 1 !important;
            display: flex !important;
            height: 100% !important;
        }
        /* Textarea occupies full height of block (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child textarea {
            flex: 1 !important;
            height: 100% !important;
            min-height: 100% !important;
            resize: none !important;
        }
        /* Align checkbox and button to right edge in one column (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:last-child {
            justify-content: flex-end !important;
            align-items: flex-end !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:last-child .block {
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-end !important;
            gap: 10px !important;
        }
        /* Improve display of phoneme sequences */
        /* Phoneme containers should use full available width */
        .gradio-container div[data-block-id='side-by-side-comparison'] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        /* Phoneme sequences should be distributed across full width */
        .gradio-container div[data-block-id='side-by-side-comparison'] > div > div {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        /* Natural wrapping of phoneme sequences, like regular text */
        /* Phoneme containers should use natural line wrapping, like regular text */
        .gradio-container div[data-block-id='side-by-side-comparison'] div[style*="font-size: 18px"][style*="line-height: 1.3"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        /* Inline-block elements should wrap naturally, like regular text */
        .gradio-container div[data-block-id='side-by-side-comparison'] div[style*="font-size: 18px"] > span[style*="display: inline-block"] {
            white-space: normal !important;
            overflow-wrap: break-word !important;
        }
        /* Horizontal scrolling for long sequences */
        .gradio-container div[style*="overflow-x: auto"] {
            scrollbar-width: thin !important;
            scrollbar-color: #cbd5e0 #f7fafc !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar {
            height: 6px !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar-track {
            background: #f7fafc !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar-thumb {
            background: #cbd5e0 !important;
            border-radius: 3px !important;
        }
        /* Adaptive distribution for different screen sizes */
        @media (min-width: 1200px) {
            .gradio-container div[data-block-id='side-by-side-comparison'] {
                max-width: calc(100vw - 200px) !important;
            }
        }
        @media (max-width: 768px) {
            .gradio-container div[data-block-id='side-by-side-comparison'] {
                max-width: calc(100vw - 40px) !important;
            }
        }
        /* Stretch main Row with chat and controls */
        .gradio-container .row.unequal-height {
            display: flex !important;
            align-items: stretch !important;
            min-height: 600px !important;
        }
        /* Stretch left column with chat */
        .gradio-container .row.unequal-height > .column:first-child {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        /* Stretch right column to chat height */
        .gradio-container .row.unequal-height > .column:nth-child(2) {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            align-items: stretch !important;
        }
        /* Stretch inner Row with controls by height */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row {
            display: flex !important;
            flex: 1 !important;
            align-items: stretch !important;
            min-height: 0 !important;
        }
        /* Align Audio Input to top edge */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:first-child {
            display: flex !important;
            align-items: flex-start !important;
        }
        /* Align validation controls to bottom edge - stretch to full height */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            align-items: stretch !important;
        }
        /* Block with validation controls - aligns button to bottom */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child > .block {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            gap: 10px !important;
            flex: 1 !important;
        }
        /* Form with validation controls - aligns button to bottom */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child > .form {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            flex: 1 !important;
        }
        /* Record button styling - make it larger and center text */
        .gradio-container button.record.record-button {
            min-width: 120px !important;
            width: auto !important;
            height: 36px !important;
            padding: 0 16px !important;
            text-align: center !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }
    """
