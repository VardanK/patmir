// Paste into the YouTube tab's console (F12 -> Console), then let an ad play.
// Logs once a second for 60s. Copy a few lines from DURING the ad.
(() => {
  const S = [".ytp-skip-ad-button", ".ytp-ad-skip-button-modern",
             ".ytp-ad-skip-button", ".ytp-ad-skip-button-slot button"];
  const n = 60; let i = 0;
  const t = setInterval(() => {
    const p = document.querySelector("#movie_player");
    const v = document.querySelector("video");
    const skip = S.map(s => [s, document.querySelector(s)]).find(([, e]) => e);
    const box = skip && skip[1].getBoundingClientRect();
    console.log("[patmir-probe]", JSON.stringify({
      cls: p ? p.className : null,
      dur: v && Math.round(v.duration),
      t: v && Math.round(v.currentTime),
      skipSel: skip ? skip[0] : null,
      skipDisabled: skip ? !!skip[1].disabled : null,
      skipBox: box ? [Math.round(box.width), Math.round(box.height)] : null,
      skipOffsetParent: skip ? (skip[1].offsetParent ? "yes" : "NULL") : null,
      adModule: !!document.querySelector(".ytp-ad-module"),
      adOverlay: !!document.querySelector(".ytp-ad-player-overlay, .ytp-ad-player-overlay-layout"),
      preview: !!document.querySelector(".ytp-ad-preview-container"),
    }));
    if (++i >= n) clearInterval(t);
  }, 1000);
  console.log("[patmir-probe] running for 60s");
})();
