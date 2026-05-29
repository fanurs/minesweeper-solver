// Minesweeper Mirror — MV3 service worker.
//
// Stateless and message-driven. All recording state, timers, and the .msm
// buffer live in the CONTENT SCRIPT (the service worker is ephemeral — Chrome
// terminates it after ~30 s idle). The worker's only future job is to issue the
// `chrome.downloads.download` call when the user triggers an Export. Empty for
// now — see PLAN.md / docs/recording-lifecycle.md (Persistence & Export).

export {};
