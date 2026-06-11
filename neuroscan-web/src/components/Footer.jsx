import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.disclaimer}>
          <svg className={styles.warnIcon} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
          <span>
            <strong>Medical Disclaimer:</strong> NeuroScan is intended for research and educational purposes only.
            It is <em>not</em> a certified medical device and must not be used for clinical diagnosis.
            All results should be reviewed by qualified medical professionals.
          </span>
        </div>
        <p className={styles.copy}>NeuroScan v1.0 &mdash; CNN-Based Brain Tumor MRI Classification &mdash; Academic / Research Use</p>
      </div>
    </footer>
  );
}
