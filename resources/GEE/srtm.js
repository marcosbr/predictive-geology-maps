/*
Script para gerar o SRTM de uma área a partir de uma shapefile.

Autor: Rodrigo Soares
Serviço Geológico do Brasil/CPRM
*/

// Caminho do dataset SRTM no servidor do GEE.
var satelite = "USGS/SRTMGL1_003"; 

/*Caminho da shape na nossa conta (aba "Assets").
Copiar o caminho do arquivo shape da área desejada e substituir na linha abaixo.*/
var shape = "users/marcosferreira/Norte_AM"; 

var area = ee.FeatureCollection(shape).geometry();

var srtm_area = ee.Image(satelite).clip(area);
      
Map.centerObject(area, 8);
Map.addLayer(srtm_area, {}, "SRTM");


Export.image.toDrive({
  image: srtm_area,
  description: "srtm_area",
  folder: "cursoMapaPreditivo", // Escrever o nome da Pasta do Google Drive que deseja salvar o arquivo utilizando "nomePasta".
  maxPixels: 1e13,
  scale: 30,
  region: area,
});

