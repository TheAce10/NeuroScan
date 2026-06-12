import { useEffect, useRef } from 'react';
import styles from './ConfidenceRing.module.css';

const CLASS_META = {
  glioma:     { label: 'Glioma',      color: '#dc2626', severity: 'High Risk' },
  meningioma: { label: 'Meningioma',  color: '#d97706', severity: 'Moderate' },
  pituitary:  { label: 'Pituitary',   color: '#2563eb', severity: 'Moderate' },
  notumor:    { label: 'No Tumor',    color: '#16a34a', severity: 'Healthy'  },
};

function normaliseKey(cls) {
  return cls.toLowerCase().replace(/[^a-z]/g, '');
}

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function ConfidenceRing({ prediction, confidence }) {
  const circleRef = useRef(null);
  const key = normaliseKey(prediction ?? '');
  const meta = CLASS_META[key] ?? { label: prediction, color: 'var(--blue-500)', severity: '' };
  const pct = Math.round((confidence ?? 0) * 100);
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;

  useEffect(() => {
    const el = circleRef.current;
    if (!el) return;
    el.style.strokeDashoffset = CIRCUMFERENCE;
    const raf = requestAnimationFrame(() => {
      el.style.transition = 'stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)';
      el.style.strokeDashoffset = offset;
    });
    return () => cancelAnimationFrame(raf);
  }, [offset]);

  return (
    <div className={styles.wrapper}>
      <div className={styles.ringWrap}>
        <svg className={styles.svg} viewBox="0 0 128 128">
          <circle className={styles.track} cx="64" cy="64" r={RADIUS} />
          <circle
            ref={circleRef}
            className={styles.fill}
            cx="64" cy="64" r={RADIUS}
            stroke={meta.color}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE}
          />
        </svg>
        <div className={styles.center}>
          <span className={styles.pct}>{pct}%</span>
          <span className={styles.confLabel}>confidence</span>
        </div>
      </div>

      <div className={styles.info}>
        <div className={styles.classLabel} style={{ color: meta.color }}>
          {meta.label}
        </div>
        <div className={styles.severity} style={{ background: `${meta.color}18`, color: meta.color, borderColor: `${meta.color}40` }}>
          {meta.severity}
        </div>
      </div>
    </div>
  );
}
