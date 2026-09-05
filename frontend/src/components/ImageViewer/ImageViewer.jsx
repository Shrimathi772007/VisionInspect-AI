import { useCallback, useEffect, useRef, useState } from "react";
import { ZoomIn, ZoomOut, Maximize } from "lucide-react";
import styles from "./ImageViewer.module.css";

const MIN_SCALE = 1;
const MAX_SCALE = 4;
const SCALE_STEP = 0.5;

function clampScale(value) {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));
}

export function ImageViewer({ src, alt }) {
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragState = useRef(null);
  const stageRef = useRef(null);

  const reset = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  const applyScale = useCallback((next) => {
    setScale((prev) => {
      const clamped = clampScale(next(prev));
      if (clamped === 1) setTranslate({ x: 0, y: 0 });
      return clamped;
    });
  }, []);

  const zoomIn = useCallback(() => applyScale((prev) => prev + SCALE_STEP), [applyScale]);
  const zoomOut = useCallback(() => applyScale((prev) => prev - SCALE_STEP), [applyScale]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;

    const handleWheel = (event) => {
      event.preventDefault();
      applyScale((prev) => prev + (event.deltaY < 0 ? SCALE_STEP : -SCALE_STEP));
    };

    // React's synthetic onWheel is registered as a passive listener, so
    // preventDefault() inside it silently fails (and warns) instead of
    // stopping the page from scrolling underneath the zoom gesture.
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [applyScale]);

  const handlePointerDown = useCallback(
    (event) => {
      if (scale <= 1) return;
      dragState.current = {
        startX: event.clientX,
        startY: event.clientY,
        originX: translate.x,
        originY: translate.y,
      };
      setIsDragging(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [scale, translate]
  );

  const handlePointerMove = useCallback((event) => {
    if (!dragState.current) return;
    const { startX, startY, originX, originY } = dragState.current;
    setTranslate({ x: originX + (event.clientX - startX), y: originY + (event.clientY - startY) });
  }, []);

  const handlePointerUp = useCallback((event) => {
    dragState.current = null;
    setIsDragging(false);
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomIn();
      } else if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        zoomOut();
      } else if (event.key === "0") {
        event.preventDefault();
        reset();
      }
    },
    [zoomIn, zoomOut, reset]
  );

  const zoomPercent = Math.round(scale * 100);

  return (
    <div className={styles.viewer}>
      <div className={styles.grid} aria-hidden="true" />
      <div className={styles.bracket} data-corner="tl" aria-hidden="true" />
      <div className={styles.bracket} data-corner="tr" aria-hidden="true" />
      <div className={styles.bracket} data-corner="bl" aria-hidden="true" />
      <div className={styles.bracket} data-corner="br" aria-hidden="true" />

      <div
        ref={stageRef}
        className={`${styles.stage} ${isDragging ? styles.dragging : ""}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onDoubleClick={reset}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        role="group"
        aria-label="Inspection image viewer. Use the zoom controls or plus, minus and zero keys."
        style={{ cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
      >
        <img
          src={src}
          alt={alt}
          className={styles.image}
          draggable={false}
          style={{ transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})` }}
        />
      </div>

      <div className={styles.toolbar}>
        <button type="button" className={styles.toolButton} onClick={zoomOut} aria-label="Zoom out" disabled={scale <= MIN_SCALE}>
          <ZoomOut size={15} />
        </button>
        <span className={styles.zoomReadout}>{zoomPercent}%</span>
        <button type="button" className={styles.toolButton} onClick={zoomIn} aria-label="Zoom in" disabled={scale >= MAX_SCALE}>
          <ZoomIn size={15} />
        </button>
        <span className={styles.divider} aria-hidden="true" />
        <button type="button" className={styles.toolButton} onClick={reset} aria-label="Reset zoom to fit">
          <Maximize size={14} />
        </button>
      </div>
    </div>
  );
}
