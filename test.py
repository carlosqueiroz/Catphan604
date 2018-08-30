from CatPhan.catphan import CatPhan604

zip_file = r"D:/CatPhan/File/1807017-fullscan1.zip"
mycbct = CatPhan604.from_zip(zip_file)
mycbct.analyze()
# print results to the console
print(mycbct.return_results())
mycbct.plot_analyzed_image()

