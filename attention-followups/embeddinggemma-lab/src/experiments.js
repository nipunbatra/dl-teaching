export const experiments = [
  {
    id: "photos",
    group: "Find",
    name: "Words → pictures",
    title: "Find a picture without naming the file",
    description:
      "Describe what you want to see. Which image would you expect to rank first?",
    mode: "search",
    type: "text",
    query: "a pet resting at home",
    filter: "image",
    ideas: [
      "a pet resting at home",
      "something that produces electricity",
      "a drink served hot",
      "a person ready for space travel",
    ],
    lesson:
      "Only pixels were embedded for the images. Their filenames and captions never entered the image encoder.",
  },
  {
    id: "sounds",
    group: "Find",
    name: "Words → sounds",
    title: "Can words find a sound?",
    description:
      "Play a recording, then try describing the sound in a different way.",
    mode: "search",
    type: "text",
    query: "a dog barking",
    filter: "audio",
    ideas: [
      "a dog barking",
      "water washing onto a beach",
      "a ticking clock",
      "a machine cutting wood",
    ],
    lesson:
      "The audio input is a waveform. There is no intermediate speech transcript in this lab.",
  },
  {
    id: "listen",
    group: "Find",
    name: "Sound → pictures & words",
    title: "Use a bark as the query",
    description:
      "Listen first. Can the recording retrieve a dog without any text query?",
    mode: "search",
    type: "audio",
    sample: "sound-dog",
    filter: "image",
    lesson:
      "Both queries and candidates live in the same 768-dimensional space. Start with image candidates, then try captions or everything. Raw cosine ranges can differ across modalities.",
  },
  {
    id: "captions",
    group: "Find",
    name: "Picture → captions",
    title: "Give the image a caption menu",
    description:
      "Choose an image. The model ranks the supplied captions; it does not write a new caption.",
    mode: "search",
    type: "image",
    sample: "chelsea",
    filter: "caption",
    lesson:
      "EmbeddingGemma 2 compares the image with every supplied caption. Open a result to inspect both vectors and their cosine similarity.",
  },
  {
    id: "neighbors",
    group: "Find",
    name: "Picture → pictures",
    title: "What makes two images similar?",
    description:
      "Will a dog photo find the same animal, the same colour, or a sketch?",
    mode: "search",
    type: "image",
    sample: "retriever-photo",
    filter: "image",
    lesson:
      "Similarity is learned. A neighbour can share subject, style or background; inspect what the model actually chose.",
  },
  {
    id: "languages",
    group: "Find",
    name: "Search in another language",
    title: "Does the query have to be in English?",
    description:
      "Keep the gallery fixed. Ask for a cat in Hindi, then try your own language.",
    mode: "search",
    type: "text",
    query: "एक बिल्ली की तस्वीर",
    filter: "image",
    ideas: [
      "एक बिल्ली की तस्वीर",
      "કોફીનો કપ",
      "un perro",
      "une fusée dans le ciel",
    ],
    lesson:
      "The text changes language; the image vectors stay fixed. Compare the top results with the equivalent English query.",
  },
  {
    id: "mixed",
    group: "Find",
    name: "Picture + words",
    title: "Two inputs, one embedding",
    description:
      "Combine an image with a short note, then compare it with an image-only search.",
    mode: "mixed",
    type: "image",
    sample: "red-mug",
    query: "a drink for breakfast",
    filter: "all",
    lesson:
      "The image and your note are encoded together in one forward pass. This is different from averaging two independently computed embeddings.",
  },
  {
    id: "moments",
    group: "Find",
    name: "Find a video moment",
    title: "Which part of the video matches?",
    description:
      "Search short excerpts from real videos of a puppy, coffee, waves and geese, alongside the original teaching slideshow.",
    mode: "search",
    type: "text",
    query: "a rocket launching",
    filter: "moment",
    ideas: [
      "a puppy playing indoors",
      "water moving in waves",
      "a rocket launching",
      "a cat looking at the camera",
      "a cup of coffee",
    ],
    lesson:
      "Each window has its own video embedding, using one sampled frame per second without the soundtrack. The same clip can match several queries. Video frames and extracted stills are related examples, not independent evaluation data.",
  },
  {
    id: "classify",
    group: "Use",
    name: "Choose your own labels",
    title: "Turn descriptions into a classifier",
    description:
      "Change the labels and run again. There is no new classifier training.",
    mode: "classify",
    type: "image",
    sample: "newfoundland",
    filter: "all",
    labels: "a dog\na cat\na rocket\na cup of coffee",
    lesson:
      "The highest cosine wins among your supplied labels. These scores are not calibrated probabilities, and the correct answer may be missing from the menu.",
  },
  {
    id: "documents",
    group: "Use",
    name: "Ask the course",
    title: "Ask the course. Read the source.",
    description: "Find explanations in Nipun’s deep learning slides. Read the excerpt, then open the slide for its figures and full context.",
    mode: "knowledge",
    lesson: "The model retrieves passages from the course. The original slide is one click away.",
  },
  {
    id: "code",
    group: "Use",
    name: "Search course code",
    title: "Describe the code you need",
    description: "Search the actual course notebooks by what the code does. Inspect the input, the calculation and the original notebook.",
    mode: "knowledge",
    lesson: "Read the surrounding notebook before running a snippet: cells can depend on earlier definitions.",
  },
  {
    id: "delta",
    group: "Inspect",
    name: "What changed?",
    title: "Compare two images",
    description:
      "Subtract the earlier image vector from the later one. Which descriptions align with that direction?",
    mode: "delta",
    type: "image",
    sample: "portrait",
    after: "portrait-hat",
    filter: "caption",
    lesson:
      "Δ = normalize(after − before). This is an exploratory direction in embedding space, not proof of what caused the change. Swap the images: every dot-product sign reverses.",
  },
  {
    id: "clusters",
    group: "Inspect",
    name: "Group the collection",
    title: "Do different modalities group together?",
    description:
      "Group the stored embeddings without using their titles or class labels.",
    mode: "clusters",
    type: "text",
    filter: "all",
    lesson:
      "K-means uses the full selected-dimensional vectors. The plot uses PCA to show two dimensions, so its distances are only an approximation. Group numbers have no predefined meaning.",
  },
  {
    id: "explorer",
    group: "Inspect",
    name: "Explore every embedding",
    title: "Where do pictures, words and sounds meet?",
    description:
      "Choose any item, open its complete vector, and compare it with another modality. Use the map to ask questions, then check the actual cosine.",
    mode: "explorer",
    lesson:
      "PCA keeps two directions of variation. The neighbour list uses the full selected-dimensional vectors.",
  },
  {
    id: "training",
    group: "Learn",
    name: "Train a small classifier",
    title: "Teach a new task using frozen embeddings",
    description:
      "Take one gradient step at a time. Train on sounds, pictures or text. Follow the 768 features into class neurons, then test on examples the head has never seen.",
    mode: "training",
    lesson: "The encoder stays fixed. Only the classifier weights learn.",
  },
];
