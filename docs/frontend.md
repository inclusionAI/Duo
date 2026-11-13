# Frontend integration

The `demo/` application is the browser interface used to inspect offline Duo inference.
It is backend-neutral: model and checkpoint names are supplied at runtime, and
the source contains no service URL, credential, or checkpoint path.

## Preview mode

Serve the directory and add `?preview=1` to the URL:

```bash
python -m http.server 8000 --directory demo
```

Open <http://127.0.0.1:8000/?preview=1>. The preview exercises recording,
upload, queue, timeline, settings, and progress layouts without calling a model.
Generated media is intentionally unavailable in this mode.

## Backend mode

Without `?preview=1`, the application connects to API routes one directory
above the frontend. A deployment commonly serves the page at `/video/` and the
API at `/api/` on the same origin. This preserves camera permissions, cookies,
byte-range video requests, and event streaming without cross-origin setup.

The frontend first requests `GET /api/frontend-catalog` and falls back to its
local `catalog.json`. The endpoint returns model labels and Cases:

```json
{
  "models": [
    {
      "id": "duo",
      "label": "Duo",
      "detail": "Configured checkpoint",
      "sampler": "Configured by the backend"
    }
  ],
  "cases": [
    {
      "id": "case",
      "label": "Case",
      "subtitle": "",
      "description": "Assistant and scene description",
      "assistant_image": "/api/cases/case/reference-image"
    }
  ]
}
```

The catalog may include a `schema` object when an existing backend uses
different JSON field names. The frontend normalizes those names to `intent`,
`realization`, and `codec` without embedding model-specific terminology.

Only the fixed assistant reference image and system description belong in the
catalog. User media always comes from the current camera recording or uploaded
video.

## API surface

The browser uses these same-origin routes:

| Route | Purpose |
| --- | --- |
| `GET /api/frontend-catalog` | Runtime model and Case labels |
| `GET /api/video/state` | Current queue, session, chunks, and media |
| `GET /api/video/events` | Server-sent state updates |
| `POST /api/video/join` | Join the shared session queue |
| `POST /api/video/accept` | Accept the offered turn |
| `POST /api/video/leave` | Leave the session or queue |
| `POST /api/video/capture` | Reserve a recording upload |
| `POST /api/video/upload` | Upload MP4 or WebM input |
| `POST /api/video/cancel-capture` | Cancel a reserved upload |
| `POST /api/reset` | Reset committed history or the full session |
| `POST /api/model` | Select a backend-defined model ID |
| `POST /api/case` | Select a Case and fixed conditions |

State chunks use public stage fields named `intent`, `realization`, and `codec`.
Media URLs should support HTTP byte ranges so cumulative video can update
without interrupting active playback.

The API retains a few serialized `block` field names for compatibility with
existing clients and checkpoints. UI copy and new documentation call each
17-frame interaction interval a chunk.
