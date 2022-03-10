/*
Script para gerar um mosaico Landsat 8 de uma área a partir de uma shapefile.
Autor: Rodrigo Soares
Serviço Geológico do Brasil/CPRM
*/

// Máscara de Nuvem
var maskL8SR = function(image) {
    var cloudShadowBitMask = ee.Number(2).pow(3).int();
    var cloudsBitMask = ee.Number(2).pow(5).int();
    var qa = image.select('QA_PIXEL');
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).and(
              qa.bitwiseAnd(cloudsBitMask).eq(0));
    return image.updateMask(mask);
  };
  
  /*Caminho da shape na nossa conta (aba "Assets").
  Copiar o caminho do arquivo shape da área desejada e substituir na linha abaixo.*/
  var shape = "users/rodrigorouto/cursoMapaPreditivo/3norte_RR_100k";
  
  var area = ee.FeatureCollection(shape).geometry();
  
  Map.centerObject(area, 9); // Aumentar ou dominuir o número para ajustar o zoom.
  
  var ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                     .filterBounds(area)
                     .filterDate('2020-01-01','2021-12-31') // Escolher intervalo de datas de aquisição das cenas.
                     .filterMetadata('CLOUD_COVER','less_than', 50) // Escolher percentual máximo de pixels com nuvens.
                     .map(maskL8SR);
  
  print(ls8);
  
  var lsMedian = ls8.median().clip(area).select(['SR_B2','SR_B3','SR_B4','SR_B6','SR_B7']); // Escolher as bandas que deseja utilizar.
  
  Map.addLayer(lsMedian,imageVisParam, 'Mosaico LS8');
  
  
  Export.image.toDrive({
    image: lsMedian,
    description: "ls8_area",
    folder: "cursoMapaPreditivo", // Escrever o nome da Pasta do Google Drive que deseja salvar o arquivo utilizando "nomePasta".
    maxPixels: 1e13,
    scale: 30,
    region: area,
  });