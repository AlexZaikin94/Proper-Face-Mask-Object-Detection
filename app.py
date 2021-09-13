import cv2
import torch

class_names = ['', 'NoMask', 'BadMask', 'Mask']
class_colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
thresh = 0.7
exit_key = 'q'


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model.pth', map_location=device)
    model.train(False)

    cam = cv2.VideoCapture(0)
    with torch.no_grad():
        while True:
            img_raw = cam.read()[1]
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img_raw, dtype=torch.float32, device=device).permute(2, 0, 1)
            out = model.forward([img])[0]
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score < thresh:
                    continue
                xmin, ymin, xmax, ymax = box.astype(int)
                cv2.rectangle(img_raw, (xmin, ymin), (xmax, ymax), class_colors[label], label)
                cv2.putText(img_raw, f"{class_names[label]} {score:.2f}", (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[label])
            cv2.imshow(f'press "{exit_key}" to quit', img_raw[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord(exit_key):
                break
    cv2.destroyAllWindows()
    cam.release()



