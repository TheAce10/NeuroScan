import { useState } from 'react';
import styles from './ImageComparison.module.css';

export default function ImageComparison({ original, heatmap, modelName, predictedClass }) {
  const [fullscreen, setFullscreen] = useState(false);
  const heatmapSrc = heatmap ? `data:image/png;base64,${heatmap}` : null;
  const ready = !!(original && heatmapSrc);

  function handleDownload() {
    if (!ready) return;

    const W = 560, H = 560, PAD = 20, LABEL_H = 32, HEADER_H = 44;
    const canvas = document.createElement('canvas');
    canvas.width  = W * 2 + PAD * 3;
    canvas.height = HEADER_H + H + LABEL_H + PAD * 2;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Header bar
    ctx.fillStyle = '#1e3a5f';
    ctx.fillRect(0, 0, canvas.width, HEADER_H);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 15px system-ui, sans-serif';
    ctx.textAlign = 'center';
    const modelSlug = (modelName  || 'Model').replace(/_/g, ' ');
    const classLabel = (predictedClass || 'unknown').charAt(0).toUpperCase()
                     + (predictedClass || 'unknown').slice(1);
    ctx.fillText(`NeuroScan · ${modelSlug} · ${classLabel}`, canvas.width / 2, HEADER_H / 2 + 6);

    const drawSide = (imgSrc, x, label, onDone) => {
      const img = new Image();
      img.onload = () => {
        const y = HEADER_H + PAD;
        ctx.drawImage(img, x, y, W, H);

        // label strip below the image
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.fillRect(x, y + H, W, LABEL_H);
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, x + W / 2, y + H + LABEL_H / 2 + 5);

        onDone();
      };
      img.crossOrigin = 'anonymous';
      img.src = imgSrc;
    };

    let done = 0;
    const finish = () => {
      done++;
      if (done < 2) return;
      const link = document.createElement('a');
      const mSlug = (modelName || 'model').toLowerCase().replace(/[\s()]+/g, '_').replace(/[^a-z0-9_]/g, '');
      const cSlug = (predictedClass || 'unknown').toLowerCase();
      link.download = `neuroscan_${mSlug}_${cSlug}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };

    drawSide(original,    PAD,           'Original Scan',    finish);
    drawSide(heatmapSrc,  PAD * 2 + W,   'Grad-CAM Heatmap', finish);
  }

  return (
    <>
      <div className={styles.grid}>
        <Panel label="Original Scan"    src={original}    tag="INPUT"  />
        <Panel label="Grad-CAM Heatmap" src={heatmapSrc}  tag="OUTPUT"
               placeholder="Heatmap will appear here" disabled={!heatmap} />
      </div>

      {/* ── Toolbar ── */}
      <div className={styles.toolbar}>
        <button
          className={styles.toolbarBtn}
          onClick={() => setFullscreen(true)}
          disabled={!ready}
          title="View both images fullscreen"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
            <path d="M3 4.25A1.25 1.25 0 014.25 3h2.5a.75.75 0 010 1.5h-2.5v2.5a.75.75 0 01-1.5 0v-2.5zM15.75 3A1.25 1.25 0 0117 4.25v2.5a.75.75 0 01-1.5 0v-2.5h-2.5a.75.75 0 010-1.5h2.5zM3 15.75A1.25 1.25 0 014.25 17h2.5a.75.75 0 010-1.5h-2.5v-2.5a.75.75 0 00-1.5 0v2.5zM15.75 17A1.25 1.25 0 0117 15.75v-2.5a.75.75 0 00-1.5 0v2.5h-2.5a.75.75 0 000 1.5h2.5z" />
          </svg>
          Fullscreen
        </button>
        <button
          className={styles.toolbarBtn}
          onClick={handleDownload}
          disabled={!ready}
          title="Download side-by-side comparison"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
            <path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" />
            <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
          </svg>
          Download
        </button>
      </div>

      {/* ── Fullscreen lightbox ── */}
      {fullscreen && (
        <div className={styles.lightbox} onClick={() => setFullscreen(false)}>
          <div className={styles.lightboxInner} onClick={e => e.stopPropagation()}>
            <div className={styles.lightboxHeader}>
              <span>Scan Comparison</span>
              <button className={styles.closeBtn} onClick={() => setFullscreen(false)}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
                  <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
                </svg>
              </button>
            </div>
            <div className={styles.lightboxGrid}>
              <div className={styles.lightboxSide}>
                <div className={styles.lightboxLabel}>Original Scan</div>
                <img src={original}    alt="Original Scan"    className={styles.lightboxImg} />
              </div>
              <div className={styles.lightboxSide}>
                <div className={styles.lightboxLabel}>Grad-CAM Heatmap</div>
                <img src={heatmapSrc}  alt="Grad-CAM Heatmap" className={styles.lightboxImg} />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Panel({ label, src, tag, placeholder, disabled }) {
  return (
    <div className={`${styles.panel} ${disabled ? styles.panelDisabled : ''}`}>
      <div className={styles.panelHeader}>
        <span className={styles.panelLabel}>{label}</span>
        <span className={styles.tag}>{tag}</span>
      </div>
      <div className={styles.imgWrap}>
        {src ? (
          <img src={src} alt={label} className={styles.img} />
        ) : (
          <div className={styles.placeholder}>
            <svg viewBox="0 0 40 40" fill="none" width="40" height="40">
              <rect x="4" y="8" width="32" height="24" rx="2" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="14" cy="17" r="3" stroke="currentColor" strokeWidth="1.5" />
              <path d="M4 26l8-8 7 7 5-5 10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span>{placeholder}</span>
          </div>
        )}
      </div>
    </div>
  );
}
