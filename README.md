# Eye Access Cues

A browser app that tracks eye movements through the device camera and shows the matching
NLP eye accessing cue (VAK): Visual, Auditory or Kinesthetic, and whether the movement is
read as **recalling** or **constructing**.

| Eyes move to (subject's own view) | Cue | Meaning |
| --- | --- | --- |
| Up, left | **Vr** Visual remembered | Recalling images |
| Up, right | **Vc** Visual constructed | Creating images |
| Level, left | **Ar** Auditory remembered | Recalling sounds |
| Level, right | **Ac** Auditory constructed | Creating sounds |
| Down, left | **Ad** Auditory digital | Internal dialogue |
| Down, right | **K** Kinesthetic | Feelings, body sensations |

"Reversed organisation" in the Sensitivity panel swaps the sides, which NLP describes for
some left-handed people.

## Run it

It's three static files (`index.html`, `styles.css`, `app.js`). Browsers only let a page use
the camera over **HTTPS** or from **localhost**, so don't open `index.html` straight from disk.

```bash
cd eye-access-cues
python3 -m http.server 8000
# open http://localhost:8000
```

To use it on a phone, host the folder on any HTTPS static host, such as GitHub Pages,
Netlify or Cloudflare Pages.

The first start downloads MediaPipe (from jsDelivr) and the face model, about 4 MB (from
Google storage). After that, all processing stays on the device. No video is uploaded.

## Using it

1. Press **Start camera**. Pick **Front camera** or **Back camera**, or choose a device from
   the list. The Back button is disabled when the device has only one camera.
2. Have the subject look straight into the lens, then press **Calibrate centre**.
3. Ask a question (the Practice questions panel has some) and watch the colour boxes. The
   matching box fills with its colour, and the large readout names the system and mode.

The chart follows the video on screen. With the mirrored front camera, the boxes match the
subject's own left and right. With the back camera, which isn't mirrored, it is the
observer's view: the subject's left appears on your right.

## How it works

- **MediaPipe Face Landmarker** returns 478 face points, including iris centres, plus
  "blendshape" scores such as `eyeLookUpLeft` and `eyeLookOutRight`.
- **Horizontal gaze**: how far each iris sits from the middle of its eye corners, measured
  along the face's own left–right axis (so tilting the head doesn't count as eye movement),
  averaged with the eyeLookIn/Out blendshapes.
- **Vertical gaze**: the eyeLookUp/Down blendshapes, averaged with the iris offset along
  the face's up axis.
- Calibration subtracts the subject's resting position. Smoothing, a threshold box with
  hysteresis, and a short hold time stop flicker. Frames where the subject blinks are
  ignored.

Tuning constants (`IRIS_X_RANGE`, `BLEND_Y_RANGE` and others) are at the top of `app.js`.

## A caution

Eye accessing cues are a teaching model from NLP. Controlled studies have not found that eye
direction reliably shows whether someone is recalling or constructing, so present the app as
a demonstration of the concept, not a lie detector. Webcam gaze estimation is also
approximate: use good, even light, keep the face 30–60 cm from the camera, and recalibrate
after anyone moves.
