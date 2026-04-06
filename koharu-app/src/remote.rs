use anyhow::{bail, Context, Result};
use koharu_core::TextBlock;
use reqwest::multipart;
use serde_json::Value;

use crate::AppResources;

/// Calls the remote Python FastAPI server for a specific engine.
/// Returns Ok(true) if the engine was handled remotely, Ok(false) if it should fall back to local,
/// or Err if the remote engine failed.
pub async fn run_remote_engine(id: &str, res: &AppResources, page_id: &str) -> Result<bool> {
    let url = {
        let config = res.config.read().await;
        if !config.remote.enabled || config.remote.url.is_empty() {
            return Ok(false);
        }
        config.remote.url.trim_end_matches('/').to_string()
    };

    let client = reqwest::Client::new();
    let doc = res.storage.page(page_id).await?;

    let is_detector = id.contains("detector") || id == "pp-doclayout-v3";
    let is_ocr = id.contains("ocr");
    let is_inpainter = id.contains("inpaint");

    if is_detector {
        let source_img = res.storage.images.load(&doc.source)?;
        let mut img_bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut img_bytes);
        source_img.write_to(&mut cursor, image::ImageFormat::Png)?;

        let part = multipart::Part::bytes(img_bytes)
            .file_name("image.png")
            .mime_str("image/png")?;
        let form = multipart::Form::new().part("image", part);

        let endpoint = format!("{}/infer/detect", url);
        tracing::info!("Sending remote request to {}", endpoint);
        
        let resp = client.post(&endpoint)
            .multipart(form)
            .send()
            .await
            .context("failed to send request to remote inference server")?;
            
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("Remote detector failed ({}): {}", status, text);
        }

        let json_resp: Value = resp.json().await.context("failed to parse remote JSON")?;
        let blocks_val = json_resp.get("text_blocks").context("missing text_blocks array")?;
        let text_blocks: Vec<TextBlock> = serde_json::from_value(blocks_val.clone())?;

        res.storage.update_page(page_id, move |d| {
            d.text_blocks = text_blocks;
        }).await?;

        return Ok(true);
    } 
    else if is_ocr {
        let source_img = res.storage.images.load(&doc.source)?;
        let mut img_bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut img_bytes);
        source_img.write_to(&mut cursor, image::ImageFormat::Png)?;

        let part = multipart::Part::bytes(img_bytes)
            .file_name("image.png")
            .mime_str("image/png")?;

        let mut boxes = Vec::new();
        for block in &doc.text_blocks {
            boxes.push([block.x, block.y, block.width, block.height]);
        }
        let boxes_json = serde_json::to_string(&boxes)?;

        let form = multipart::Form::new()
            .part("image", part)
            .text("boxes", boxes_json);

        let endpoint = format!("{}/infer/ocr", url);
        tracing::info!("Sending remote request to {}", endpoint);
        
        let resp = client.post(&endpoint)
            .multipart(form)
            .send()
            .await
            .context("failed to send request to remote inference server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("Remote OCR failed ({}): {}", status, text);
        }

        let json_resp: Value = resp.json().await.context("failed to parse remote JSON")?;
        let texts_val = json_resp.get("texts").context("missing texts array")?;
        let texts: Vec<String> = serde_json::from_value(texts_val.clone())?;

        if texts.len() == doc.text_blocks.len() {
            res.storage.update_page(page_id, move |d| {
                for (i, text) in texts.into_iter().enumerate() {
                    d.text_blocks[i].text = Some(text);
                }
            }).await?;
        } else {
            bail!("Remote OCR returned {} results for {} blocks", texts.len(), doc.text_blocks.len());
        }

        return Ok(true);
    }
    else if is_inpainter {
        // Implementation for inpainting: needs to send image & mask, get back inpainted image.
        let source_img = res.storage.images.load(&doc.source)?;
        let mut img_bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut img_bytes);
        source_img.write_to(&mut cursor, image::ImageFormat::Png)?;

        let part = multipart::Part::bytes(img_bytes)
            .file_name("image.png")
            .mime_str("image/png")?;

        let mut mask_bytes = Vec::new();
        if let Some(mask_hash) = &doc.segment {
            let mask_img = res.storage.images.load(mask_hash)?;
            let mut cursor = std::io::Cursor::new(&mut mask_bytes);
            mask_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        }

        let mask_part = multipart::Part::bytes(mask_bytes)
            .file_name("mask.png")
            .mime_str("image/png")?;
            
        let form = multipart::Form::new()
            .part("image", part)
            .part("mask", mask_part);
            
        let endpoint = format!("{}/infer/inpaint", url);
        tracing::info!("Sending remote request to {}", endpoint);
        
        let resp = client.post(&endpoint)
            .multipart(form)
            .send()
            .await
            .context("failed to send request to remote inference server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("Remote inpaint failed ({}): {}", status, text);
        }
        
        let result_bytes = resp.bytes().await?;
        let result_img = image::load_from_memory(&result_bytes).context("failed to decode remote inpainted image")?;
        
        let inpainted_hash = res.storage.images.store_webp(&result_img)?;
        
        res.storage.update_page(page_id, move |d| {
            d.inpainted = Some(inpainted_hash);
        }).await?;

        return Ok(true);
    }

    Ok(false)
}
