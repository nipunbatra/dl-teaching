// Fixed, inspectable teaching splits. Labels supervise the head; they are never
// appended to media inputs. Paired captions use the same split as their images.
export const TRAINING_TASKS = [
  { id: "3", name: "Sounds · dog, waves, fire", type: "audio" },
  { id: "10", name: "Sounds · all 10 categories", type: "audio" },
  { id: "images", name: "Pictures · animals, food, transport", type: "image" },
  { id: "text", name: "Text · animals, food, transport", type: "text" },
];

const visualSplit = [
  { label: "Animals", train: ["newfoundland", "Persian", "commons-extra-12090096", "commons-extra-61295118"], test: ["chelsea", "commons-extra-20283834"] },
  { label: "Food & drink", train: ["coffee", "photo-pizza", "commons-extra-22552194", "commons-extra-151875501"], test: ["commons-extra-113972132", "commons-extra-9534738"] },
  { label: "Transport", train: ["rocket", "photo-train", "photo-bicycle", "commons-extra-61339647"], test: ["commons-extra-97567388", "commons-extra-153714724"] },
];
const captionId = (id) => id.startsWith("photo-")
  ? `caption-commons-${id.slice(6)}` : `caption-${id}`;

export function trainingTask(id, gallery, vectors) {
  const task = TRAINING_TASKS.find((x) => x.id === id);
  if (!task) throw new Error(`Unknown training task: ${id}`);
  if (task.type === "audio") {
    const classes = id === "3" ? ["dog", "sea_waves", "crackling_fire"]
      : [...new Set(gallery.filter((x) => x.type === "audio" && x.label).map((x) => x.label))];
    const data = gallery.filter((x) => x.type === "audio" && classes.includes(x.label) && ["train", "test"].includes(x.split));
    if (data.some((x) => !vectors[x.id])) throw new Error("A training embedding is missing.");
    return { ...task, classes, train: data.filter((x) => x.split === "train"), test: data.filter((x) => x.split === "test") };
  }
  const data = visualSplit.flatMap((group) => ["train", "test"].flatMap((split) => group[split].map((sourceId) => {
    const itemId = task.type === "text" ? captionId(sourceId) : sourceId;
    const item = gallery.find((x) => x.id === itemId);
    if (!item || !vectors[itemId]) throw new Error(`Missing training sample: ${itemId}`);
    return { ...item, label: group.label, split, sourceGroup: sourceId };
  })));
  return { ...task, classes: visualSplit.map((x) => x.label), train: data.filter((x) => x.split === "train"), test: data.filter((x) => x.split === "test") };
}

export const className = (label) => label.replaceAll("_", " ");
