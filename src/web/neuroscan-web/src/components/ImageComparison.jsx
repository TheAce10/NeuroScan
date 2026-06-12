import { useState } from 'react';
import styles from './ImageComparison.module.css';

export default function ImageComparison({ original, heatmap }) {
  const [enlarged, setEnlarged] = useState(null);

  return (
    <>
      <div className={styles.grid}>
        <Panel
          label="Original Scan"
          src={original}
          tag="INPUT"
          onEnlarge={() => setEnlarged({ src: original, label: 'Original Scan' })}
        />
        <Panel
          label="Grad-CAM Heatmap"
          src={heatmap ? `data:image/png;base64,${heatmap}` : null}
          tag="GRAD-CAM"
          placeholder="Heatmap will appear here"
          onEnlarge={() => setEnlarged({ src: `data:image/png;base64,${heatmap}`, label: 'Grad-CAM Heatmap' })}
          disabled={!heatmap}
        />
      </div>

      {enlarged && (
        <div className={styles.lightbox} onClick={() => setEnlarged(null)}>
          <div className={styles.lightboxInner} onClick={e => e.stopPropagation()}>
            <div className={styles.lightboxHeader}>
              <span>{enlarged.label}</span>
              <button className={styles.closeBtn} onClick={() => setEnlarged(null)}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
                  <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
                </svg>
              </button>
            </div>
            <img src={enlarged.src} alt={enlarged.label} className={styles.lightboxImg} />
          </div>
        </div>
      )}
    </>
  );
}

function Panel({ label, src, tag, placeholder, onEnlarge, disabled }) {
  return (
    <div className={`${styles.panel} ${disabled ? styles.panelDisabled : ''}`}>
      <div className={styles.panelHeader}>
        <span className={styles.panelLabel}>{label}</span>
        <span className={styles.tag}>{tag}</span>
      </div>
      <div className={styles.imgWrap}>
        {src ? (
          <>
            <img src={src} alt={label} className={styles.img} />
            <button className={styles.enlargeBtn} onClick={onEnlarge} title="View fullscreen">
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                <path d="M3 4.25A1.25 1.25 0 014.25 3h2.5a.75.75 0 010 1.5h-2.5v2.5a.75.75 0 01-1.5 0v-2.5zM15.75 3A1.25 1.25 0 0117 4.25v2.5a.75.75 0 01-1.5 0v-2.5h-2.5a.75.75 0 010-1.5h2.5zM3 15.75A1.25 1.25 0 014.25 17h2.5a.75.75 0 010-1.5h-2.5v-2.5a.75.75 0 00-1.5 0v2.5zM15.75 17A1.25 1.25 0 0017 15.75v-2.5a.75.75 0 00-1.5 0v2.5h-2.5a.75.75 0 000 1.5h2.5z" />
              </svg>
            </button>
          </>
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
