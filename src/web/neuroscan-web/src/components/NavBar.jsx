import { Link, useLocation } from 'react-router-dom';
import styles from './NavBar.module.css';

export default function NavBar() {
  const { pathname } = useLocation();
  const onScanner = pathname === '/scan';

  return (
    <>
      <nav className={styles.nav}>
        <div className={styles.inner}>
          <Link to="/" className={styles.brand}>
            <img src="/NeuroScan/ccre-logo.png" alt="CCRE" className={styles.logo} />
            <span className={styles.title}>
              CCRE <span className={styles.sep}>|</span> NeuroScan
            </span>
          </Link>
          <div className={styles.right}>
            <span className={styles.badge}>Research Use Only</span>
            {onScanner
              ? <Link to="/" className={styles.ctaGhost}>← Overview</Link>
              : <Link to="/scan" className={styles.cta}>Launch Scanner</Link>
            }
          </div>
        </div>
      </nav>
      <div className={styles.disclaimer}>
        For research and educational purposes only — not for clinical diagnosis or medical decision-making.
      </div>
    </>
  );
}
