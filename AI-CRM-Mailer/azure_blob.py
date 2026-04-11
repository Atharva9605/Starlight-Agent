"""
Azure Blob Storage manager for Starlight catalogue page images.

Each ingested catalogue page is rendered as a PNG and stored in Azure Blob
Storage so that email templates can embed a direct image URL alongside the
product reference.

Required .env variables:
    AZURE_STORAGE_CONNECTION_STRING  - full connection string  (preferred)
      OR
    AZURE_STORAGE_ACCOUNT_NAME  +  AZURE_STORAGE_ACCOUNT_KEY
    AZURE_STORAGE_CONTAINER          - blob container name  (default: starlight-catalogues)

If no credentials are configured the manager returns placeholder URLs so the
rest of the pipeline can still run (without real images in emails).
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("azure_blob")

CONTAINER_DEFAULT = "starlight-catalogues"

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    _BLOB_SDK_OK = True
except ImportError:
    _BLOB_SDK_OK = False
    log.warning(
        "azure-storage-blob is not installed. "
        "Run: pip install azure-storage-blob\n"
        "Blob upload will be skipped and placeholder URLs used."
    )


class AzureBlobManager:
    """
    Uploads PNG catalogue page images to Azure Blob Storage and returns
    their public HTTP URLs for use in emails and metadata.
    """

    def __init__(self) -> None:
        self.connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.account_name: str = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
        self.account_key: str = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
        self.container: str = os.getenv("AZURE_STORAGE_CONTAINER", CONTAINER_DEFAULT)
        self._client = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the SDK is installed and credentials are configured."""
        if not _BLOB_SDK_OK:
            return False
        return bool(self.connection_string) or bool(self.account_name and self.account_key)

    # ------------------------------------------------------------------
    # Internal: get (or create) client + container
    # ------------------------------------------------------------------

    def _get_service_client(self) -> "BlobServiceClient":
        if self._client is None:
            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            elif self.account_name and self.account_key:
                url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=url, credential=self.account_key
                )
            else:
                raise RuntimeError(
                    "No Azure Blob Storage credentials found. "
                    "Set AZURE_STORAGE_CONNECTION_STRING in your .env file."
                )
        return self._client

    def _ensure_container(self) -> None:
        """Create the container with public-blob access if it doesn't exist."""
        client = self._get_service_client()
        cc = client.get_container_client(self.container)
        try:
            cc.get_container_properties()
        except Exception:
            cc.create_container(public_access="blob")
            log.info("Created Azure Blob container: %s", self.container)

    def _infer_account_name(self) -> str:
        """Extract account name from connection string."""
        if self.account_name:
            return self.account_name
        for part in self.connection_string.split(";"):
            if part.startswith("AccountName="):
                return part.split("=", 1)[1]
        return "storageaccount"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        content_type: str = "image/png",
    ) -> str:
        """
        Upload a single page PNG and return its public URL.

        Args:
            image_bytes:    PNG bytes of the rendered page.
            catalogue_slug: Short slug for the catalogue filename
                            (used as the blob folder prefix).
            page_number:    1-indexed page number.
            content_type:   MIME type (default image/png).

        Returns:
            Full HTTPS URL to the blob, or a placeholder string if
            blob storage is not configured.
        """
        if not self.available:
            placeholder = (
                f"[blob_not_configured]/{catalogue_slug}/page_{page_number:03d}.png"
            )
            log.debug("Blob storage not configured. Placeholder: %s", placeholder)
            return placeholder

        blob_name = f"{catalogue_slug}/page_{page_number:03d}.png"
        try:
            self._ensure_container()
            client = self._get_service_client()
            blob_client = client.get_blob_client(
                container=self.container, blob=blob_name
            )
            blob_client.upload_blob(
                image_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            account = self._infer_account_name()
            url = (
                f"https://{account}.blob.core.windows.net"
                f"/{self.container}/{blob_name}"
            )
            log.debug("Uploaded %s → %s", blob_name, url)
            return url

        except Exception as exc:
            log.error("Blob upload failed for %s page %d: %s", catalogue_slug, page_number, exc)
            return f"[upload_failed]/{catalogue_slug}/page_{page_number:03d}.png"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
blob_manager = AzureBlobManager()
