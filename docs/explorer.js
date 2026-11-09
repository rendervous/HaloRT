// Interactive comparison explorer: click a scene, hover any image to see a
// synced zoomed crop (same source position) across all methods at once.
(function () {
    const MANIFEST_URL = "assets/explorer/manifest.json";
    const IMAGE_BASE = "assets/explorer/";
    const ZOOM_WINDOW_FRAC = 0.12; // fraction of the image's shorter side used as the zoom source window
    const ZOOM_CANVAS_SIZE = 220;  // displayed zoom canvas size, px

    const switcherEl = document.getElementById("scene-switcher");
    const rowEl = document.getElementById("explorer-row");
    const zoomRowEl = document.getElementById("zoom-row");
    const hintEl = document.getElementById("explorer-hint");

    let scenes = [];
    let currentSceneIdx = 0;
    let loadedImages = []; // Image objects for the current scene, same order as scene.methods
    let lastUV = { u: 0.5, v: 0.5 };
    let hasHovered = false;

    function labelFor(method) {
        if (method.psnr == null) return method.name;
        return `${method.name} — ${method.psnr.toFixed(2)} dB, ${method.time_ms.toFixed(1)} ms`;
    }

    function buildSwitcher() {
        switcherEl.innerHTML = "";
        scenes.forEach((scene, i) => {
            const btn = document.createElement("button");
            btn.className = "scene-btn" + (i === currentSceneIdx ? " active" : "");
            btn.textContent = scene.name.replace(/_/g, " ");
            btn.addEventListener("click", () => {
                currentSceneIdx = i;
                renderScene();
            });
            switcherEl.appendChild(btn);
        });
    }

    function updateSwitcherActive() {
        [...switcherEl.children].forEach((btn, i) => {
            btn.classList.toggle("active", i === currentSceneIdx);
        });
    }

    function drawZoom(canvas, img, u, v) {
        const ctx = canvas.getContext("2d");
        const nw = img.naturalWidth, nh = img.naturalHeight;
        if (!nw || !nh) return;
        const winPx = Math.round(ZOOM_WINDOW_FRAC * Math.min(nw, nh));
        let sx = Math.round(u * nw - winPx / 2);
        let sy = Math.round(v * nh - winPx / 2);
        sx = Math.max(0, Math.min(nw - winPx, sx));
        sy = Math.max(0, Math.min(nh - winPx, sy));
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, sx, sy, winPx, winPx, 0, 0, canvas.width, canvas.height);
    }

    function updateAllZooms(u, v) {
        lastUV = { u, v };
        const canvases = zoomRowEl.querySelectorAll("canvas");
        canvases.forEach((canvas, i) => {
            if (loadedImages[i]) drawZoom(canvas, loadedImages[i], u, v);
        });
        // crosshair markers on each main image
        rowEl.querySelectorAll(".crosshair").forEach((el) => {
            el.style.left = (u * 100) + "%";
            el.style.top = (v * 100) + "%";
            el.style.display = hasHovered ? "block" : "none";
        });
    }

    function handleMove(evt, imgEl) {
        const rect = imgEl.getBoundingClientRect();
        const u = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
        const v = Math.min(1, Math.max(0, (evt.clientY - rect.top) / rect.height));
        hasHovered = true;
        if (hintEl) hintEl.style.display = "none";
        updateAllZooms(u, v);
    }

    function renderScene() {
        const scene = scenes[currentSceneIdx];
        updateSwitcherActive();
        rowEl.innerHTML = "";
        zoomRowEl.innerHTML = "";
        loadedImages = [];

        scene.methods.forEach((method) => {
            const cell = document.createElement("div");
            cell.className = "explorer-cell";

            const imgWrap = document.createElement("div");
            imgWrap.className = "explorer-img-wrap";

            const img = new Image();
            img.src = IMAGE_BASE + method.image;
            img.alt = method.name;
            img.className = "explorer-img";
            img.addEventListener("mousemove", (evt) => handleMove(evt, img));
            loadedImages.push(img);

            const crosshair = document.createElement("div");
            crosshair.className = "crosshair";

            imgWrap.appendChild(img);
            imgWrap.appendChild(crosshair);

            const caption = document.createElement("div");
            caption.className = "explorer-caption";
            caption.textContent = labelFor(method);

            cell.appendChild(imgWrap);
            cell.appendChild(caption);
            rowEl.appendChild(cell);

            const zoomCell = document.createElement("div");
            zoomCell.className = "zoom-cell";
            const canvas = document.createElement("canvas");
            canvas.width = ZOOM_CANVAS_SIZE;
            canvas.height = ZOOM_CANVAS_SIZE;
            const zoomCaption = document.createElement("div");
            zoomCaption.className = "explorer-caption";
            zoomCaption.textContent = method.name;
            zoomCell.appendChild(canvas);
            zoomCell.appendChild(zoomCaption);
            zoomRowEl.appendChild(zoomCell);

            img.addEventListener("load", () => {
                if (hasHovered) drawZoom(canvas, img, lastUV.u, lastUV.v);
                else drawZoom(canvas, img, 0.5, 0.42);
            });
        });

        if (hintEl) hintEl.style.display = hasHovered ? "none" : "block";
    }

    fetch(MANIFEST_URL)
        .then((r) => r.json())
        .then((data) => {
            scenes = data;
            buildSwitcher();
            renderScene();
        })
        .catch((err) => {
            rowEl.innerHTML = "<p class='note'>Could not load the explorer data (" + err + ").</p>";
        });
})();
