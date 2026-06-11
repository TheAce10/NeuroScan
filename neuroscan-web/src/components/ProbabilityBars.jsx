import { useEffect, useRef } from 'react';
import styles from './ProbabilityBars.module.css';

const CLASSES = [
  { key: 'glioma',     label: 'Glioma',     color: '#dc2626' },
  { key: 'meningioma', label: 'Meningioma', color: '#d97706' },
  { key: 'pituitary',  label: 'Pituitary',  color: '#2563eb' },
  { key: 'notumor',    label: 'No Tumor',   color: '#16a34a' },
];

function normaliseScores(scores) {
  if (!scores) return {};
  const out = {};
  for (const [k, v] of Object.entries(scores)) {
    out[k.toLowerCase().replace(/[^a-z]/g, '')] = v;
  }
  return out;
}

export default function ProbabilityBars({ scores, prediction }) {
  const barRefs = useRef({});
  const normalised = normaliseScores(scores);
  const predKey = (prediction ?? '').toLowerCase().replace(/[^a-z]/g, '');

  useEffect(() => {
    CLASSES.forEach(({ key }) => {
      const el = barRefs.current[key];
      if (!el) return;
      el.style.width = '0%';
      const raf = requestAnimationFrame(() => {
        el.style.transition = 'width 0.9s cubic-bezier(.4,0,.2,1)';
        el.style.width = `${Math.round((normalised[key] ?? 0) * 100)}%`;
      });
      return () => cancelAnimationFrame(raf);
    });
  }, [scores]);

  return (
    <div className={styles.wrapper}>
      <h3 className={styles.heading}>Class Probabilities</h3>
      <div className={styles.bars}>
        {CLASSES.map(({ key, label, color }) => {
          const pct = Math.round((normalised[key] ?? 0) * 100);
          const isTop = key === predKey;
          return (
            <div key={key} className={`${styles.row} ${isTop ? styles.topClass : ''}`}>
              <div className={styles.meta}>
                <span className={styles.label}>{label}</span>
                <span className={styles.pct} style={{ color }}>{pct}%</span>
              </div>
              <div className={styles.track}>
                <div
                  ref={el => { barRefs.current[key] = el; }}
                  className={styles.fill}
                  style={{ background: color, width: '0%' }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
