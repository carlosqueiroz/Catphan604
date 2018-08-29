from CatPhan.catphan import CatPhan604

zip_file = r"D:/CatPhan/File/1807017-fullscan1.zip"
mycbct = CatPhan604.from_zip(zip_file)
mycbct.analyze()
# print results to the console
print(mycbct.return_results())
"""
mycbct.write_csv('mtf_b.csv')
# view analyzed images
mycbct.plot_center_hu_profile()
mycbct.plot_center_hu('hu_profile')
mycbct.plot_analyzed_image()
mycbct._return_results()
mycbct.plot_analyzed_subimage(subimage='hu')
mycbct.plot_analyzed_subimage(subimage='un')
mycbct.plot_analyzed_subimage(subimage='sp')
mycbct.plot_analyzed_subimage(subimage='lc')
mycbct.save_analyzed_subimage('lc.png', subimage='lc')
mycbct.save_analyzed_subimage('sp.png', subimage='sp')
mycbct.plot_single_frame()
# save the image
mycbct.save_analyzed_image('mycatphan503.png')
"""
