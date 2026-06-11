import { useState } from 'react';
import styles from './ApiSettings.module.css';

const STATUS = { idle: 'idle', checking: 'checking', ok: 'ok', error: 'error' };

export default function ApiSettings({ endpoint, onChange, onModels }) {
  const [status, setStatus] = useState(STATUS.idle);
  const [info, setInfo] = useState(null);

  async function checkHealth() {
    setStatus(STATUS.checking);
    setInfo(null);
    try {
      const res = await fetch(`${endpoint.replace(/\/$/, '')}/health`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setInfo(data);
      setStatus(STATUS.ok);
      if (onModels && Array.isArray(data.models)) onModels(data.models);
    } catch (err) {
      setStatus(STATUS.error);
    }
  }

  const dot = {
    [STATUS.idle]:     styles.dotIdle,
    [STATUS.checking]: styles.dotChecking,
    [STATUS.ok]:       styles.dotOk,
    [STATUS.error]:    styles.dotError,
  }[status];

  const statusText = {
    [STATUS.idle]:     'Not checked',
    [STATUS.checking]: 'Checking…',
    [STATUS.ok]:       `Online${info?.device ? ` · ${info.device.toUpperCase()}` : ''}`,
    [STATUS.error]:    'Unreachable',
  }[status];

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <svg className={styles.icon} viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zM2 10a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 10zm0 5.25a.75.75 0 01.75-.75H10a.75.75 0 010 1.5H2.75a.75.75 0 01-.75-.75z" clipRule="evenodd" />
        </svg>
        <span className={styles.title}>API Endpoint</span>
      </div>

      <div className={styles.body}>
        <div className={styles.inputRow}>
          <input
            className={styles.input}
            type="text"
            value={endpoint}
            onChange={e => { onChange(e.target.value); setStatus(STATUS.idle); setInfo(null); }}
            placeholder="http://localhost:8000"
            spellCheck={false}
          />
          <button
            className={styles.checkBtn}
            onClick={checkHealth}
            disabled={status === STATUS.checking || !endpoint}
          >
            {status === STATUS.checking ? 'Checking…' : 'Check'}
          </button>
        </div>

        <div className={styles.statusRow}>
          <span className={`${styles.dot} ${dot}`} />
          <span className={styles.statusText}>{statusText}</span>
          {status === STATUS.ok && info?.classes && (
            <span className={styles.classes}>{info.classes.join(' · ')}</span>
          )}
        </div>
      </div>
    </div>
  );
}
