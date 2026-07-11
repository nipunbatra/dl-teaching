---
title: "Lecture 10: Localization, Detection and Segmentation"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 10: Localization, Detection and Segmentation

## Core story

\[
\boxed{
\text{Classification predicts one label. Dense vision predicts structure.}
}
\]

Tasks:

\[
\text{classification}
\rightarrow
\text{localization}
\rightarrow
\text{object detection}
\rightarrow
\text{segmentation}
\]

Object detection has two major families: two-stage region-based systems such as R-CNN/Faster R-CNN, and one-stage dense predictors such as YOLO. R-CNN combined region proposals with CNN features; Faster R-CNN made proposals neural via an RPN; YOLO reframed detection as direct regression from full images to boxes and class probabilities.

---

## Slide-level outline

### Part I — Task taxonomy

1. **Title**  
   *Localization, Object Detection and Segmentation*

2. **Classification**
   \[
   x\rightarrow y
   \]

3. **Localization**
   \[
   x\rightarrow (y,b)
   \]
   one dominant object + box.

4. **Object detection**
   \[
   x\rightarrow \{(y_i,b_i,s_i)\}_{i=1}^{N}
   \]

5. **Semantic segmentation**
   \[
   x\rightarrow y_{hw}
   \quad\forall h,w
   \]

6. **Instance segmentation**
   \[
   x\rightarrow \{(y_i,b_i,m_i,s_i)\}
   \]

7. **Panoptic segmentation**
   every pixel gets:
   \[
   (\text{class},\text{instance id})
   \]

8. **Visual comparison slide**
   Same image with:
   - class label;
   - box;
   - many boxes;
   - semantic mask;
   - instance masks.

---

### Part II — Bounding boxes and localization

9. **Box parameterizations**
   \[
   b=(x_{\min},y_{\min},x_{\max},y_{\max})
   \]
   or:
   \[
   b=(x_c,y_c,w,h)
   \]

10. **Normalize coordinates**
   \[
   x_c/W,\quad y_c/H,\quad w/W,\quad h/H
   \]

11. **Localization network**
   CNN backbone:
   \[
   h=f_\theta(x)
   \]
   Classification head:
   \[
   p=\operatorname{softmax}(W_ch)
   \]
   Box head:
   \[
   \hat b=W_bh
   \]

12. **Multi-task loss**
   \[
   L=L_{\text{cls}}+\lambda L_{\text{box}}
   \]

13. **Box regression loss**
   Simple:
   \[
   L_{\text{box}}=\|\hat b-b\|_1
   \]
   or:
   \[
   \|\hat b-b\|_2^2
   \]

14. **Smooth \(L_1\)**
   \[
   \operatorname{smooth}_{L_1}(r)=
   \begin{cases}
   0.5r^2,&|r|<1\\
   |r|-0.5,&\text{otherwise}
   \end{cases}
   \]

15. **Worked localization example**
   True:
   \[
   b=(0.5,0.5,0.4,0.2)
   \]
   Pred:
   \[
   \hat b=(0.6,0.45,0.5,0.3)
   \]
   Compute:
   \[
   L_1=0.1+0.05+0.1+0.1=0.35
   \]

16. **Interactive 1: bbox loss**
   `bbox-loss.html`

---

### Part III — IoU and detection metrics

17. **Intersection over Union**
   \[
   \operatorname{IoU}(A,B)=
   \frac{|A\cap B|}{|A\cup B|}
   \]

18. **Worked IoU example**
   Box A area \(=100\), Box B area \(=100\), intersection \(=25\).
   \[
   \text{union}=100+100-25=175
   \]
   \[
   \text{IoU}=25/175=0.143
   \]

19. **Why IoU loss?**
   Coordinate losses do not directly optimize overlap.

20. **GIoU idea**
   IoU has zero-gradient problems when boxes do not overlap; GIoU was proposed as both a metric and loss to address this weakness.

21. **Precision and recall**
   \[
   \text{precision}=\frac{TP}{TP+FP}
   \]
   \[
   \text{recall}=\frac{TP}{TP+FN}
   \]

22. **Average Precision**
   area under precision-recall curve.

23. **mAP**
   mean AP across classes and IoU thresholds.

24. **Interactive 2: IoU/AP**
   `iou-map-demo.html`

---

### Part IV — Object detection pipeline

25. **Detection challenges**
   - unknown number of objects;
   - unknown locations;
   - different scales;
   - class imbalance;
   - duplicate predictions.

26. **Dense candidate idea**
   Predict boxes at many locations/scales.

27. **Anchor boxes**
   Predefined reference boxes:
   \[
   a=(x_a,y_a,w_a,h_a)
   \]

28. **Box offsets**
   \[
   t_x=\frac{x-x_a}{w_a}
   \]
   \[
   t_y=\frac{y-y_a}{h_a}
   \]
   \[
   t_w=\log\frac{w}{w_a}
   \]
   \[
   t_h=\log\frac{h}{h_a}
   \]

29. **Detection head outputs**
   For each anchor/location:
   - objectness;
   - class logits;
   - box offsets.

30. **Detection loss**
   \[
   L=
   L_{\text{obj}}
   +
   L_{\text{cls}}
   +
   \lambda L_{\text{box}}
   \]

31. **Positive/negative assignment**
   Anchor positive if:
   \[
   \operatorname{IoU}(a,b_{\text{gt}})>\tau_+
   \]
   negative if:
   \[
   \operatorname{IoU}(a,b_{\text{gt}})<\tau_-
   \]

32. **Class imbalance**
   Most anchors are background.

33. **Focal loss**
   \[
   FL(p_t)=-(1-p_t)^\gamma\log p_t
   \]
   Focal loss was designed for dense one-stage detection to down-weight easy examples and address foreground-background imbalance.

34. **Interactive 3: anchor assignment**
   `anchor-assignment.html`

---

### Part V — R-CNN family

35. **R-CNN**
   Pipeline:
   ```text
   region proposals → crop/warp → CNN features → classifier + box regressor
   ```
   R-CNN used region proposals with CNN features and showed a large mAP improvement on PASCAL VOC.

36. **Limitations**
   - many crops;
   - slow;
   - multi-stage training.

37. **Fast R-CNN**
   ```text
   image → CNN feature map → RoI pooling → heads
   ```
   Fast R-CNN improved training/testing speed by sharing convolution over the image.

38. **Faster R-CNN**
   Adds Region Proposal Network:
   ```text
   image → backbone → RPN proposals → RoI head
   ```
   Faster R-CNN introduced an RPN sharing full-image convolutional features with the detector.

39. **RPN loss**
   \[
   L_{\text{RPN}}=L_{\text{objectness}}+\lambda L_{\text{box}}
   \]

40. **RoI head loss**
   \[
   L_{\text{RoI}}=L_{\text{cls}}+\lambda L_{\text{box}}
   \]

41. **Two-stage detector intuition**
   Stage 1:
   \[
   \text{Where might objects be?}
   \]
   Stage 2:
   \[
   \text{What object and exact box?}
   \]

---

### Part VI — YOLO / one-stage detectors

42. **YOLO idea**
   ```text
   image → one network → boxes + classes
   ```
   YOLO framed detection as direct regression to bounding boxes and class probabilities from the full image in one evaluation.

43. **Grid intuition**
   Divide image into grid cells; each predicts boxes/classes.

44. **One-stage detector loss**
   \[
   L=L_{\text{box}}+L_{\text{obj}}+L_{\text{cls}}
   \]

45. **Speed vs accuracy intuition**
   - one-stage: simpler/faster;
   - two-stage: historically stronger localization/accuracy;
   - modern differences depend on implementation, data and scale.

46. **Non-maximum suppression**
   Problem: many boxes for same object.

47. **NMS algorithm**
   1. sort boxes by score;
   2. keep highest;
   3. suppress boxes with IoU \(>\tau\);
   4. repeat.

48. **Interactive 4: NMS**
   `nms-demo.html`

---

### Part VII — Segmentation

49. **Semantic segmentation**
   Per-pixel classification:
   \[
   p_{hwk}=p(y_{hw}=k\mid x)
   \]

50. **Pixelwise cross-entropy**
   \[
   L=
   -\frac1{HW}
   \sum_{h,w}
   \log p_{hw,y_{hw}}
   \]

51. **Fully convolutional networks**
   FCNs converted classification networks into dense predictors and combined coarse semantic with shallow appearance information for segmentation.

52. **Encoder-decoder**
   ```text
   image → downsample/context → upsample/pixel prediction
   ```

53. **U-Net**
   Contracting path captures context; symmetric expanding path enables precise localization with skip connections.

54. **Dice coefficient**
   \[
   \operatorname{Dice}
   =
   \frac{2|P\cap G|}{|P|+|G|}
   \]

55. **Soft Dice loss**
   \[
   L_{\text{Dice}}
   =
   1-
   \frac{2\sum_i p_ig_i+\epsilon}
   {\sum_i p_i+\sum_i g_i+\epsilon}
   \]

56. **When Dice helps**
   Severe foreground-background imbalance.

57. **Instance segmentation**
   Detection + mask per object.

58. **Mask R-CNN**
   Extends Faster R-CNN with a parallel mask-prediction branch.

59. **Mask loss**
   Per-pixel binary cross-entropy inside RoI:
   \[
   L_{\text{mask}}
   =
   -\sum_{u,v}
   \left[
   m_{uv}\log \hat m_{uv}
   +(1-m_{uv})\log(1-\hat m_{uv})
   \right]
   \]

60. **Interactive 5: segmentation losses**
   `segmentation-losses.html`

---

### Part VIII — Summary

61. **Task/loss table**

| Task | Output | Main loss |
|---|---|---|
| classification | class | CE |
| localization | class + one box | CE + box loss |
| detection | many boxes/classes | obj + cls + box |
| semantic segmentation | class per pixel | pixel CE / Dice |
| instance segmentation | boxes + masks | detection + mask loss |

62. **Detector family table**

| Family | Idea |
|---|---|
| R-CNN | proposals + CNN |
| Fast R-CNN | shared features + RoI head |
| Faster R-CNN | neural proposals via RPN |
| YOLO | direct dense one-stage prediction |
| Mask R-CNN | detection + instance masks |

63. **Final mental model**
   \[
   \boxed{
   \text{classification: what}
   }
   \]
   \[
   \boxed{
   \text{localization/detection: what + where}
   }
   \]
   \[
   \boxed{
   \text{segmentation: what at each pixel}
   }
   \]

---

## Advanced CV interactives

1. `bbox-loss.html`
2. `iou-map-demo.html`
3. `anchor-assignment.html`
4. `nms-demo.html`
5. `segmentation-losses.html`
6. `detection-pipeline.html`

## Advanced CV notebooks

1. `01_bbox_iou_losses.ipynb`
2. `02_map_precision_recall.ipynb`
3. `03_anchor_assignment.ipynb`
4. `04_nms_from_scratch.ipynb`
5. `05_toy_yolo_loss.ipynb`
6. `06_semantic_segmentation_unet_toy.ipynb`
7. `07_dice_vs_ce_loss.ipynb`

My suggestion: make **CNN one full lecture**, and make advanced CV either **one dense survey lecture** or better **two lectures**: detection first, segmentation second.
