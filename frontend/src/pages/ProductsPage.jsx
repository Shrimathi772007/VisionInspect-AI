import { useMemo, useState } from "react";
import { Plus, Search, Box, Hash, Calendar, ScanEye } from "lucide-react";
import { useProducts } from "../hooks/useProducts";
import { useInspections } from "../hooks/useInspections";
import { createProduct } from "../api/products";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Button } from "../components/Button/Button";
import { Input } from "../components/Input/Input";
import { Modal } from "../components/Modal/Modal";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { Badge } from "../components/Badge/Badge";
import { formatDateTime } from "../utils/formatDate";
import styles from "./ProductsPage.module.css";

export function ProductsPage() {
  const { products, isLoading, error, refetch } = useProducts();
  const { inspections } = useInspections();
  const { showToast } = useToast();

  const [searchTerm, setSearchTerm] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formValues, setFormValues] = useState({ productName: "", productCode: "" });
  const [formErrors, setFormErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const inspectionCountByProduct = useMemo(() => {
    const counts = new Map();
    for (const inspection of inspections) {
      counts.set(inspection.product_id, (counts.get(inspection.product_id) || 0) + 1);
    }
    return counts;
  }, [inspections]);

  const filteredProducts = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return products;
    return products.filter(
      (p) => p.product_name.toLowerCase().includes(term) || p.product_code.toLowerCase().includes(term)
    );
  }, [products, searchTerm]);

  const openModal = () => {
    setFormValues({ productName: "", productCode: "" });
    setFormErrors({});
    setIsModalOpen(true);
  };

  const closeModal = () => {
    if (isSubmitting) return;
    setIsModalOpen(false);
  };

  const validate = () => {
    const errors = {};
    if (!formValues.productName.trim()) errors.productName = "Product name is required.";
    if (!formValues.productCode.trim()) errors.productCode = "Product code is required.";
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) return;

    setIsSubmitting(true);
    try {
      await createProduct(formValues);
      showToast({ type: "success", title: "Product created", message: `${formValues.productName} was added.` });
      setIsModalOpen(false);
      refetch();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setFormErrors({ productCode: "This product code is already in use." });
      } else {
        showToast({
          type: "error",
          title: "Couldn't create product",
          message: err instanceof ApiError ? err.message : "Something went wrong.",
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Product lines"
        title="Products"
        description="Register the product lines that inspections will be captured against."
        actions={
          <Button leftIcon={<Plus size={16} />} onClick={openModal}>
            Create Product
          </Button>
        }
      />

      {products.length > 0 && (
        <div className={styles.searchRow}>
          <Input
            placeholder="Search by name or product code..."
            leftIcon={<Search size={16} />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search products"
          />
        </div>
      )}

      {isLoading && (
        <div className={styles.grid}>
          {[1, 2, 3, 4].map((key) => (
            <Card key={key} className={styles.productCard}>
              <Skeleton variant="circle" width={40} height={40} />
              <Skeleton width="70%" height={16} style={{ marginTop: 16 }} />
              <Skeleton width="45%" height={12} style={{ marginTop: 8 }} />
            </Card>
          ))}
        </div>
      )}

      {!isLoading && error && (
        <Card className={styles.errorCard}>
          <p>Failed to load products. {error.message}</p>
          <Button size="sm" variant="secondary" onClick={refetch}>
            Retry
          </Button>
        </Card>
      )}

      {!isLoading && !error && products.length === 0 && (
        <Card>
          <EmptyState
            icon={Box}
            title="No products yet"
            description="Create your first product line to start capturing inspections against it."
            action={
              <Button leftIcon={<Plus size={16} />} onClick={openModal}>
                Create Product
              </Button>
            }
          />
        </Card>
      )}

      {!isLoading && !error && products.length > 0 && filteredProducts.length === 0 && (
        <Card>
          <EmptyState icon={Search} title="No matches" description={`No products match "${searchTerm}".`} />
        </Card>
      )}

      {!isLoading && !error && filteredProducts.length > 0 && (
        <div className={styles.grid}>
          {filteredProducts.map((product) => (
            <Card key={product.id} hoverable className={styles.productCard}>
              <div className={styles.productIcon}>
                <Box size={18} />
              </div>
              <h3 className={styles.productName}>{product.product_name}</h3>
              <div className={styles.productMetaRow}>
                <Hash size={12} />
                <span className={styles.productCode}>{product.product_code}</span>
              </div>
              <div className={styles.productFooter}>
                <span className={styles.footerItem}>
                  <Calendar size={12} />
                  {formatDateTime(product.created_at)}
                </span>
                <Badge tone="neutral" dot>
                  <ScanEye size={11} style={{ marginRight: -2 }} />
                  {inspectionCountByProduct.get(product.id) || 0}
                </Badge>
              </div>
            </Card>
          ))}
        </div>
      )}

      <Modal
        open={isModalOpen}
        onClose={closeModal}
        title="Create product"
        description="Add a new product line to associate inspections with."
        footer={
          <>
            <Button variant="ghost" onClick={closeModal} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button onClick={handleSubmit} loading={isSubmitting}>
              Create Product
            </Button>
          </>
        }
      >
        <form className={styles.form} onSubmit={handleSubmit}>
          <Input
            label="Product name"
            placeholder="e.g. Aluminum Bracket A12"
            value={formValues.productName}
            onChange={(e) => setFormValues((v) => ({ ...v, productName: e.target.value }))}
            error={formErrors.productName}
          />
          <Input
            label="Product code"
            placeholder="e.g. ABK-A12"
            value={formValues.productCode}
            onChange={(e) => setFormValues((v) => ({ ...v, productCode: e.target.value }))}
            error={formErrors.productCode}
            helperText={!formErrors.productCode ? "Must be unique across all products." : undefined}
          />
        </form>
      </Modal>
    </div>
  );
}
