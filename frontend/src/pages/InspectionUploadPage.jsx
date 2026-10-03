import { useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, AlertCircle, Box, ArrowRight, RotateCcw, Plus, Info, TriangleAlert } from "lucide-react";
import { useProducts } from "../hooks/useProducts";
import { uploadInspection } from "../api/inspections";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Select } from "../components/Select/Select";
import { Button } from "../components/Button/Button";
import { FileDropzone } from "../components/FileDropzone/FileDropzone";
import { ImagePreview } from "../components/ImagePreview/ImagePreview";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { Badge } from "../components/Badge/Badge";
import { categoryLabel } from "../constants/mvtecCategories";
import { aiPredictionLabel, aiPredictionTone } from "../utils/badgeMaps";
import styles from "./InspectionUploadPage.module.css";

export function InspectionUploadPage() {
  const { products, isLoading: productsLoading } = useProducts();
  const { showToast } = useToast();

  const [productId, setProductId] = useState("");
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [createdInspection, setCreatedInspection] = useState(null);

  const selectedProduct = products.find((product) => String(product.id) === String(productId));

  const handleFileAccepted = (accepted) => {
    setFile(accepted);
    setFileError(null);
  };

  const handleReset = () => {
    setFile(null);
    setFileError(null);
    setUploadError(null);
    setCreatedInspection(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setUploadError(null);

    if (!productId) {
      setUploadError("Select a product before uploading.");
      return;
    }
    if (!file) {
      setUploadError("Choose an image to upload.");
      return;
    }

    setIsUploading(true);
    try {
      const inspection = await uploadInspection({ productId, file });
      setCreatedInspection(inspection);
      showToast({ type: "success", title: "Inspection created", message: `Inspection #${inspection.id} was recorded.` });
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  if (createdInspection) {
    return (
      <div>
        <PageHeader eyebrow="Image acquisition" title="Upload Inspection" />
        <Card className={styles.successCard}>
          <div className={styles.successIcon}>
            <CheckCircle2 size={28} />
          </div>
          <h2 className={styles.successTitle}>Inspection recorded</h2>
          <p className={styles.successDescription}>
            Inspection <strong>#{createdInspection.id}</strong> has been uploaded and stored successfully.
          </p>
          <div className={styles.aiSummary}>
            {createdInspection.ai_prediction ? (
              <>
                <span>AI result:</span>
                <Badge tone={aiPredictionTone(createdInspection.ai_prediction)}>
                  {aiPredictionLabel(createdInspection.ai_prediction)}
                </Badge>
              </>
            ) : typeof createdInspection.product_category === "string" ? (
              <span>
                No AI result: no AI model is available for the {categoryLabel(createdInspection.product_category)}{" "}
                category yet.
              </span>
            ) : createdInspection.product_category === null ? (
              <span>No AI result: this product has no MVTec category.</span>
            ) : null}
          </div>
          <div className={styles.successActions}>
            <Button as={Link} to={`/inspections/${createdInspection.id}`} rightIcon={<ArrowRight size={16} />}>
              View Inspection
            </Button>
            <Button variant="secondary" leftIcon={<RotateCcw size={16} />} onClick={handleReset}>
              Upload Another
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="Image acquisition"
        title="Upload Inspection"
        description="Attach an inspection image to a product to create a new inspection record."
      />

      {!productsLoading && products.length === 0 ? (
        <Card>
          <EmptyState
            icon={Box}
            title="Create a product first"
            description="You need at least one product before you can upload an inspection image."
            action={
              <Button as={Link} to="/products" leftIcon={<Plus size={16} />}>
                Create Product
              </Button>
            }
          />
        </Card>
      ) : (
        <Card className={styles.formCard}>
          <form onSubmit={handleSubmit} className={styles.form}>
            {productsLoading ? (
              <Skeleton height={44} />
            ) : (
              <Select
                label="Product"
                value={productId}
                onChange={(e) => setProductId(e.target.value)}
                required
              >
                <option value="" disabled>
                  Select a product
                </option>
                {products.map((product) => (
                  <option key={product.id} value={product.id}>
                    {product.product_name} ({product.product_code}) -{" "}
                    {product.category ? categoryLabel(product.category) : "No category"}
                  </option>
                ))}
              </Select>
            )}

            {selectedProduct && (
              <p className={styles.categoryNote} aria-live="polite">
                <Info size={14} aria-hidden="true" />
                <span>
                  {selectedProduct.category
                    ? `AI analysis will run using the ${categoryLabel(selectedProduct.category)} model if one is available.`
                    : "No category set. The image will be stored but not analysed by AI. Ask a Quality Engineer to set a category on this product."}
                </span>
              </p>
            )}

            <p className={styles.cautionNote}>
              <TriangleAlert size={14} aria-hidden="true" />
              <span>
                Upload an image of the same product type as the selected category. A different object type gives
                unreliable results.
              </span>
            </p>

            <div className={styles.fileField}>
              <span className={styles.fileFieldLabel}>Inspection image</span>
              {file ? (
                <ImagePreview file={file} onRemove={() => setFile(null)} disabled={isUploading} />
              ) : (
                <FileDropzone onFileAccepted={handleFileAccepted} onFileRejected={setFileError} disabled={isUploading} />
              )}
              {fileError && (
                <p className={styles.fieldError} role="alert">
                  <AlertCircle size={13} /> {fileError}
                </p>
              )}
            </div>

            {uploadError && (
              <div className={styles.uploadErrorBanner} role="alert">
                <AlertCircle size={16} />
                <span>{uploadError}</span>
              </div>
            )}

            {isUploading && (
              <div className={styles.progressWrap} aria-live="polite">
                <div className={styles.progressTrack}>
                  <div className={styles.progressBar} />
                </div>
                <span className={styles.progressLabel}>Uploading image&hellip;</span>
              </div>
            )}

            <Button type="submit" size="lg" fullWidth loading={isUploading} disabled={!productId || !file}>
              Create Inspection
            </Button>
          </form>
        </Card>
      )}
    </div>
  );
}
