// Feature comparison slider: drag to reveal HaloRT's final reconstruction
// (left) vs. one of its own input features (right, picked via buttons).
// "Random scene" jumps to a different example from the same scene set.
(function () {
    const MANIFEST_URL = "assets/feature_explorer/manifest.json";
    const IMAGE_BASE = "assets/feature_explorer/";

    const switcherEl = document.getElementById("feature-switcher");
    const randomBtn = document.getElementById("feature-random-btn");
    const sliderEl = document.getElementById("feature-slider");
    const baseImgEl = document.getElementById("feature-base-img");
    const topImgEl = document.getElementById("feature-top-img");
    const dividerEl = document.getElementById("feature-divider");
    const labelRightEl = document.getElementById("feature-label-right");
    const sceneNameEl = document.getElementById("feature-scene-name");

    let scenes = [];
    let currentSceneIdx = 0;
    let currentFeature = null; // feature name (e.g. "depth"), set once the manifest loads
    let dragging = false;

    function featureList(scene) {
        return scene.features.filter((f) => f.name !== "reconstruction");
    }

    function findFeature(scene, name) {
        return scene.features.find((f) => f.name === name);
    }

    function setSplit(pct) {
        pct = Math.min(100, Math.max(0, pct));
        topImgEl.style.clipPath = `inset(0 0 0 ${pct}%)`;
        dividerEl.style.left = pct + "%";
    }

    function updateImages() {
        const scene = scenes[currentSceneIdx];
        const recon = findFeature(scene, "reconstruction");
        const feat = findFeature(scene, currentFeature) || featureList(scene)[0];
        baseImgEl.src = IMAGE_BASE + recon.image;
        baseImgEl.alt = "Reconstruction";
        topImgEl.src = IMAGE_BASE + feat.image;
        topImgEl.alt = feat.label;
        labelRightEl.textContent = feat.label;
        sceneNameEl.textContent = scene.name.replace(/_/g, " ");
    }

    function buildFeatureSwitcher() {
        const scene = scenes[currentSceneIdx];
        switcherEl.innerHTML = "";
        featureList(scene).forEach((feat) => {
            const btn = document.createElement("button");
            btn.className = "scene-btn" + (feat.name === currentFeature ? " active" : "");
            btn.textContent = feat.label;
            btn.dataset.feature = feat.name;
            btn.addEventListener("click", () => {
                currentFeature = feat.name;
                updateFeatureSwitcherActive();
                updateImages();
            });
            switcherEl.appendChild(btn);
        });
    }

    function updateFeatureSwitcherActive() {
        [...switcherEl.children].forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.feature === currentFeature);
        });
    }

    function render() {
        if (currentFeature == null) {
            currentFeature = featureList(scenes[currentSceneIdx])[0].name;
        }
        buildFeatureSwitcher();
        updateImages();
        setSplit(50);
    }

    function pickRandomScene() {
        if (scenes.length < 2) return;
        let idx;
        do {
            idx = Math.floor(Math.random() * scenes.length);
        } while (idx === currentSceneIdx);
        currentSceneIdx = idx;
        render();
    }

    function posFromEvent(evt) {
        const rect = sliderEl.getBoundingClientRect();
        const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
        return ((clientX - rect.left) / rect.width) * 100;
    }

    function startDrag(evt) {
        dragging = true;
        setSplit(posFromEvent(evt));
        evt.preventDefault();
    }

    function moveDrag(evt) {
        if (!dragging) return;
        setSplit(posFromEvent(evt));
    }

    function endDrag() {
        dragging = false;
    }

    sliderEl.addEventListener("mousedown", startDrag);
    window.addEventListener("mousemove", moveDrag);
    window.addEventListener("mouseup", endDrag);
    sliderEl.addEventListener("touchstart", startDrag, { passive: false });
    window.addEventListener("touchmove", moveDrag, { passive: false });
    window.addEventListener("touchend", endDrag);

    randomBtn.addEventListener("click", pickRandomScene);

    fetch(MANIFEST_URL)
        .then((r) => r.json())
        .then((data) => {
            scenes = data;
            currentSceneIdx = Math.floor(Math.random() * scenes.length);
            render();
        })
        .catch((err) => {
            sliderEl.innerHTML = "<p class='note'>Could not load the feature data (" + err + ").</p>";
        });
})();
