//! UMAP and Barnes-Hut t-SNE built on scalerank's kNN, PCA and random
//! streams, so the browser app and the research scripts can compare
//! ScaleRank with them on the same footing. Not part of the published
//! `scalerank` crate or Python package.

pub mod tsne;
pub mod umap;
