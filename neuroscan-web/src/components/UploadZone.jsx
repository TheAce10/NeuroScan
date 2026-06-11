import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styles from './UploadZone.module.css';

const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/webp': [] };
const MAX_SIZE = 10 * 1024 * 1024; // 10 MB

export default function UploadZone({ onFile, isLoading }) {
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');

  const onDrop = useCallback((accepted, rejected) => {
    setError('');
    if (rejected.length > 0) {
      const msg = rejected[0].errors[0]?.message ?? 'Invalid file.';
      setError(msg);
      return;
    }
    if (accepted.length === 0) return;
    const file = accepted[0];
    const url = URL.createObjectURL(file);
    setPreview(url);
    onFile(file);
  }, [onFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: isLoading,
  });

  const handleClear = (e) => {
    e.stopPropagation();
    setPreview(null);
    setError('');
    onFile(null);
  };

  return (
    <div className={styles.wrapper}>
      <div
        {...getRootProps()}
        className={`${styles.zone} ${isDragActive ? styles.active : ''} ${isLoading ? styles.disabled : ''} ${preview ? styles.hasPreview : ''}`}
      >
        <input {...getInputProps()} />

        {preview ? (
          <div className={styles.previewWrap}>
            <img src={preview} alt="Uploaded MRI scan" className={styles.previewImg} />
            {!isLoading && (
              <button className={styles.clearBtn} onClick={handleClear} title="Remove image">
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
                  <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
                </svg>
              </button>
            )}
          </div>
        ) : (
          <div className={styles.placeholder}>
            <svg className={styles.uploadIcon} viewBox="0 0 48 48" fill="none" aria-hidden="true">
              <rect x="6" y="10" width="36" height="28" rx="3" stroke="currentColor" strokeWidth="2" />
              <circle cx="18" cy="20" r="4" stroke="currentColor" strokeWidth="2" />
              <path d="M6 32l10-10 8 8 6-6 12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className={styles.primary}>
              {isDragActive ? 'Drop the MRI scan here' : 'Drag & drop an MRI scan'}
            </p>
            <p className={styles.secondary}>or <span className={styles.link}>click to browse</span></p>
            <p className={styles.hint}>JPEG · PNG · WebP &mdash; max 10 MB</p>
          </div>
        )}

        {isLoading && (
          <div className={styles.loadingOverlay}>
            <div className={styles.spinner} />
            <span>Analysing scan&hellip;</span>
          </div>
        )}
      </div>

      {error && (
        <p className={styles.error}>
          <svg viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
            <path d="M8 1a7 7 0 100 14A7 7 0 008 1zM7.25 4.75a.75.75 0 011.5 0v3.5a.75.75 0 01-1.5 0v-3.5zM8 11a1 1 0 110-2 1 1 0 010 2z" />
          </svg>
          {error}
        </p>
      )}
    </div>
  );
}
