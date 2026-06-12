import styles from './Header.module.css';

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <img src="/NeuroScan/ccre-logo.png" alt="CCRE" className={styles.icon} />
          <span className={styles.title}>NeuroScan</span>
          <span className={styles.subtitle}>Brain Tumor MRI Classification</span>
        </div>
        <div className={styles.badge}>Research Use Only</div>
      </div>
    </header>
  );
}
