import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Database, ImageOff, Search, UploadCloud, CheckCircle2 } from "lucide-react";
import { useDatasetCategories } from "../hooks/useDatasetCategories";
import { useProducts } from "../hooks/useProducts";
import {
  getDatasetCategoryDetail,
  listDatasetImages,
  getDatasetPreviewObjectUrl,
  importDatasetInspection,
} from "../api/dataset";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { Select } from "../components/Select/Select";
import { Modal } from "../components/Modal/Modal";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import styles from "./DatasetBrowserPage.module.css";

function DatasetThumb({ category, split, defectType, filename, onClick }) {
  const [url, setUrl] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let objectUrl = null;
    setUrl(null);
    setFailed(false);

    getDatasetPreviewObjectUrl({ category, split, defectType, filename, maxDim: 200 })
      .then((result) => {
        if (cancelled) {
          URL.revokeObjectURL(result);
          return;
        }
        objectUrl = result;
        setUrl(result);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [category, split, defectType, filename]);

  return (
    <button type="button" className={styles.thumb} onClick={onClick}>
      {url && <img src={url} alt={filename} className={styles.thumbImage} />}
      {!url && !failed && <Skeleton variant="block" height="100%" />}
      {failed && (
        <div className={styles.thumbError}>
          <ImageOff size={18} />
        </div>
      )}
      <span className={styles.thumbFilename}>{filename}</span>
    </button>
  );
}

export function DatasetBrowserPage() {
  const { categories, isLoading: categoriesLoading, error: categoriesError } = useDatasetCategories();
  const { products, isLoading: productsLoading } = useProducts();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [selectedCategory, setSelectedCategory] = useState(null);
  const [categoryDetail, setCategoryDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [selectedSplit, setSelectedSplit] = useState("test");
  const [selectedDefectType, setSelectedDefectType] = useState(null);

  const [images, setImages] = useState([]);
  const [imagesLoading, setImagesLoading] = useState(false);

  const [importTarget, setImportTarget] = useState(null);
  const [importProductId, setImportProductId] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!selectedCategory && categories.length > 0) {
      setSelectedCategory(categories[0]);
    }
  }, [categories, selectedCategory]);

  useEffect(() => {
    if (!selectedCategory) return;
    let cancelled = false;
    setDetailLoading(true);
    setCategoryDetail(null);

    getDatasetCategoryDetail(selectedCategory)
      .then((data) => {
        if (cancelled) return;
        setCategoryDetail(data);
        const testSplit = data.splits.find((s) => s.split === "test");
        const defaultDefect =
          testSplit?.defect_types.find((d) => d.defect_type === "good") || testSplit?.defect_types[0];
        setSelectedSplit("test");
        setSelectedDefectType(defaultDefect?.defect_type || null);
      })
      .catch(() => {
        if (!cancelled) setCategoryDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedCategory]);

  useEffect(() => {
    if (!selectedCategory || !selectedSplit || !selectedDefectType) return;
    let cancelled = false;
    setImagesLoading(true);
    setImages([]);

    listDatasetImages({ category: selectedCategory, split: selectedSplit, defectType: selectedDefectType })
      .then((data) => {
        if (!cancelled) setImages(data.filenames);
      })
      .catch(() => {
        if (!cancelled) setImages([]);
      })
      .finally(() => {
        if (!cancelled) setImagesLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedCategory, selectedSplit, selectedDefectType]);

  const currentSplitDefectTypes = categoryDetail?.splits.find((s) => s.split === selectedSplit)?.defect_types || [];

  const openImportModal = async (filename) => {
    setImportTarget({ filename });
    setImportProductId("");
    setPreviewUrl(null);
    try {
      const url = await getDatasetPreviewObjectUrl({
        category: selectedCategory,
        split: selectedSplit,
        defectType: selectedDefectType,
        filename,
        maxDim: 480,
      });
      setPreviewUrl(url);
    } catch {
      // A failed larger preview isn't fatal — the import flow can proceed without it.
    }
  };

  const closeImportModal = () => {
    if (isImporting) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImportTarget(null);
    setPreviewUrl(null);
  };

  const handleImport = async () => {
    if (!importProductId || !importTarget) return;
    setIsImporting(true);
    try {
      const inspection = await importDatasetInspection({
        productId: importProductId,
        category: selectedCategory,
        split: selectedSplit,
        defectType: selectedDefectType,
        filename: importTarget.filename,
      });
      showToast({
        type: "success",
        title: "Inspection imported",
        message: `Inspection #${inspection.id} was created from ${selectedCategory}/${selectedDefectType}/${importTarget.filename}.`,
      });
      closeImportModal();
      navigate(`/inspections/${inspection.id}`);
    } catch (err) {
      showToast({
        type: "error",
        title: "Import failed",
        message: err instanceof ApiError ? err.message : "Something went wrong.",
      });
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="MVTec AD dataset"
        title="Dataset Browser"
        description="Browse the MVTec anomaly detection categories and import sample images as inspections."
      />

      {categoriesLoading && (
        <div className={styles.chipRow}>
          {[1, 2, 3, 4, 5].map((key) => (
            <Skeleton key={key} width={90} height={32} style={{ borderRadius: 999 }} />
          ))}
        </div>
      )}

      {!categoriesLoading && categoriesError && (
        <Card className={styles.errorCard}>
          <p>Failed to load dataset categories. {categoriesError.message}</p>
        </Card>
      )}

      {!categoriesLoading && !categoriesError && categories.length === 0 && (
        <Card>
          <EmptyState
            icon={Database}
            title="No dataset categories found"
            description="The MVTec AD dataset could not be found on the server."
          />
        </Card>
      )}

      {!categoriesLoading && categories.length > 0 && (
        <>
          <div className={styles.chipRow} role="tablist" aria-label="Dataset categories">
            {categories.map((category) => (
              <button
                key={category}
                type="button"
                role="tab"
                aria-selected={category === selectedCategory}
                className={`${styles.chip} ${category === selectedCategory ? styles.chipActive : ""}`}
                onClick={() => setSelectedCategory(category)}
              >
                {category}
              </button>
            ))}
          </div>

          {detailLoading && (
            <Card className={styles.controlsCard}>
              <Skeleton height={32} />
            </Card>
          )}

          {!detailLoading && categoryDetail && (
            <Card className={styles.controlsCard}>
              <div className={styles.splitRow}>
                {["train", "test"].map((split) => (
                  <button
                    key={split}
                    type="button"
                    className={`${styles.splitButton} ${split === selectedSplit ? styles.splitButtonActive : ""}`}
                    onClick={() => setSelectedSplit(split)}
                  >
                    {split === "train" ? "Train" : "Test"}
                  </button>
                ))}
              </div>

              <div className={styles.defectRow}>
                {currentSplitDefectTypes.map((d) => (
                  <button
                    key={d.defect_type}
                    type="button"
                    className={`${styles.defectChip} ${d.defect_type === selectedDefectType ? styles.defectChipActive : ""}`}
                    onClick={() => setSelectedDefectType(d.defect_type)}
                  >
                    <Badge tone={d.defect_type === "good" ? "success" : "danger"} dot>
                      {d.defect_type}
                    </Badge>
                    <span className={styles.defectCount}>{d.count}</span>
                  </button>
                ))}
              </div>
            </Card>
          )}

          {imagesLoading && (
            <div className={styles.grid}>
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} variant="block" height={140} />
              ))}
            </div>
          )}

          {!imagesLoading && currentSplitDefectTypes.length > 0 && images.length === 0 && (
            <Card>
              <EmptyState icon={Search} title="No images" description="This bucket has no images." />
            </Card>
          )}

          {!imagesLoading && images.length > 0 && (
            <div className={styles.grid}>
              {images.map((filename) => (
                <DatasetThumb
                  key={filename}
                  category={selectedCategory}
                  split={selectedSplit}
                  defectType={selectedDefectType}
                  filename={filename}
                  onClick={() => openImportModal(filename)}
                />
              ))}
            </div>
          )}
        </>
      )}

      <Modal
        open={Boolean(importTarget)}
        onClose={closeImportModal}
        title={importTarget ? `${selectedCategory} / ${selectedDefectType} / ${importTarget.filename}` : ""}
        description="Preview this MVTec AD image and optionally import it as an inspection record."
        footer={
          <RoleGate allow={["quality_engineer"]}>
            <Button variant="ghost" onClick={closeImportModal} disabled={isImporting}>
              Close
            </Button>
            <Button
              onClick={handleImport}
              loading={isImporting}
              disabled={!importProductId || productsLoading}
              leftIcon={<UploadCloud size={16} />}
            >
              Import as Inspection
            </Button>
          </RoleGate>
        }
      >
        <div className={styles.previewWrap}>
          {previewUrl ? (
            <img src={previewUrl} alt={importTarget?.filename} className={styles.previewImage} />
          ) : (
            <Skeleton variant="block" height={280} />
          )}
        </div>

        <RoleGate
          allow={["quality_engineer"]}
          fallback={
            <p className={styles.viewOnlyNote}>
              <CheckCircle2 size={14} /> Viewing as Factory Supervisor. Only Quality Engineers can import images.
            </p>
          }
        >
          {productsLoading ? (
            <Skeleton height={44} />
          ) : products.length === 0 ? (
            <p className={styles.viewOnlyNote}>
              Create a product first from the <Link to="/products">Products</Link> page before importing.
            </p>
          ) : (
            <Select label="Product" value={importProductId} onChange={(e) => setImportProductId(e.target.value)} required>
              <option value="" disabled>
                Select a product
              </option>
              {products.map((product) => (
                <option key={product.id} value={product.id}>
                  {product.product_name} ({product.product_code})
                </option>
              ))}
            </Select>
          )}
        </RoleGate>
      </Modal>
    </div>
  );
}
