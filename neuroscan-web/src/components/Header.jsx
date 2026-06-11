import styles from './Header.module.css';

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <svg className={styles.icon} viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <circle cx="16" cy="16" r="15" stroke="currentColor" strokeWidth="2" />
            <path
              d="M9 16c0-3.866 3.134-7 7-7s7 3.134 7 7"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round"
            />
            <circle cx="16" cy="16" r="3" fill="currentColor" />
            <path d="M16 9V7M9.5 12.5l-1.5-1.5M22.5 12.5l1.5-1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <span className={styles.title}>NeuroScan</span>
          <span className={styles.subtitle}>Brain Tumor MRI Classification</span>
        </div>
        <div className={styles.badge}>Research Use Only</div>
      </div>
    </header>
  );
}
