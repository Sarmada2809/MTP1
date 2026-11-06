/*
 * STEP 3 - Run this in the Google Earth Engine Code Editor (code.earthengine.google.com)
 *
 * BEFORE RUNNING
 *  1. Zip all files of data_processed/gee_upload/WB_grid_5km.* (shp, shx, dbf, prj, cpg).
 *  2. Earth Engine > Assets > NEW > Table Upload > Shapefiles, select the zip. Wait until done.
 *  3. Copy the asset ID and paste it below.
 *
 * WHAT IT DOES
 *  For every 5 km cell it computes
 *    - share of each land-cover class (ESA WorldCover 2021, 8 classes like the paper)
 *    - mean built-up surface (GHSL 2020), mean elevation (SRTM), mean NDVI (MODIS 2020)
 *    - total population (WorldPop 2020)
 *  and exports two CSVs to your Google Drive. Download both into data_raw/.
 */
var GRID_ASSET = 'projects/YOUR_PROJECT/assets/WB_grid_5km';   // <-- CHANGE THIS
var grid = ee.FeatureCollection(GRID_ASSET);
Map.centerObject(grid, 7);
Map.addLayer(grid, {color: 'red'}, 'grid');

// ---------- land cover: fraction of each class ----------
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var lc = ee.Image.cat([
  wc.eq(40).rename('lc_crop'),
  wc.eq(10).rename('lc_tree'),
  wc.eq(30).rename('lc_grass'),
  wc.eq(20).rename('lc_shrub'),
  wc.eq(90).or(wc.eq(95)).rename('lc_wetland'),   // herbaceous wetland + mangroves
  wc.eq(50).rename('lc_built'),
  wc.eq(60).rename('lc_bare'),
  wc.eq(80).rename('lc_water')
]).toFloat();

// ---------- extra layers (our extension beyond the paper) ----------
var built = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
              .select('built_surface').rename('built_surface');
var elev = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation');
var ndvi = ee.ImageCollection('MODIS/061/MOD13Q1')
             .filterDate('2020-01-01', '2021-01-01')
             .select('NDVI').mean().multiply(0.0001).rename('ndvi');

var meanImg = lc.addBands(built).addBands(elev).addBands(ndvi);
var meanNames = ['lc_crop','lc_tree','lc_grass','lc_shrub','lc_wetland',
                 'lc_built','lc_bare','lc_water','built_surface','elevation','ndvi'];

var meanFC = meanImg.reduceRegions({
  collection: grid,
  reducer: ee.Reducer.mean(),
  scale: 30,          // if you get a memory error, change to 100
  tileScale: 16
});

Export.table.toDrive({
  collection: meanFC,
  description: 'WB_gee_means',
  fileNamePrefix: 'WB_gee_means',
  fileFormat: 'CSV',
  selectors: ['Grid_ID'].concat(meanNames)
});

// ---------- population: SUM of people per cell ----------
var popCol = ee.ImageCollection('WorldPop/GP/100m/pop')
               .filter(ee.Filter.eq('country', 'IND'))
               .filter(ee.Filter.eq('year', 2020));
var popProj = popCol.first().projection();
var pop = popCol.mosaic().setDefaultProjection(popProj).rename('pop_sum');

var popFC = pop.reduceRegions({
  collection: grid,
  reducer: ee.Reducer.sum().setOutputs(['pop_sum']),
  scale: popProj.nominalScale(),
  tileScale: 4
});

Export.table.toDrive({
  collection: popFC,
  description: 'WB_gee_pop',
  fileNamePrefix: 'WB_gee_pop',
  fileFormat: 'CSV',
  selectors: ['Grid_ID', 'pop_sum']
});

print('Now open the Tasks tab (top right) and click RUN on both exports.');
