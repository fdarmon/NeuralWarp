#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

from pathlib import Path

def pack_folder_to_html(folder_path):
    name = "visu"
    folder = Path(folder_path)
    html_path = folder.parent / (name + ".html")

    with open(html_path, 'w') as f:
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('\t<title></title>\n')
        f.write('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
        f.write('\t<meta name=\"robots\" content=\"index, follow\" />\n')
        f.write('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
        f.write('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
        f.write('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
        f.write('\t<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" media=\"screen\" />\n')
        f.write('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
        f.write("\t<style>\n"
                     "\ttable {\n"
                     "\t\tborder-collapse: collapse;\n"
                     "\t}\n\t</style>\n")

        f.write('</head>\n')
        f.write('<body>\n')
        f.write("\t<table>\n")

        l = list(folder.glob("depth_*.jpg"))
        iterations = sorted([int(t.name.split(".")[0].split("_")[1]) for t in l])

        for iteration in iterations:

            if not (folder / f"gt_view0_{iteration}.jpg").exists():
                break

            f.write("\t<tr>\n")

            # ==========================================================================================================
            #  Ground truth images
            # ==========================================================================================================

            j = 0
            while True:
                im_name = f"gt_view{j}_{iteration}.jpg"
                if not (folder / im_name).exists():
                    break
                im_path_rel = f"{folder_path.name}/gt_view{j}_{iteration}.jpg"
                msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" width=\"400\" /></a> </td>\n'.format(
                im_path_rel,
                im_path_rel,
                im_path_rel)
                f.write(msg)
                j += 1

            f.write("\t</tr>\n")
            f.write("\t<tr>\n")

            # ==========================================================================================================
            #  Depth / normals / ...
            # ==========================================================================================================

            # depth image
            im_path_rel = f"{folder_path.name}/depth_{iteration}.jpg"
            msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" width=\"400\"/></a> </td>\n'.format(
            im_path_rel,
            im_path_rel,
            im_path_rel)
            f.write(msg)

            # normal image
            im_name = f"normal_{iteration}.jpg"
            im_path_rel = f"{folder_path.name}/normal_{iteration}.jpg"
            if (folder / im_name).exists():
                msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" width=\"400\"/></a> </td>\n'.format(
                im_path_rel,
                im_path_rel,
                im_path_rel)
                f.write(msg)


            # ==========================================================================================================
            #  Renderings
            # ==========================================================================================================

            f.write("\t</tr>\n")
            f.write("\t<tr>\n")

            # network rendering
            im_name = f"network_rendering_{iteration}.gif"
            im_path_rel = f"{folder_path.name}/network_rendering_{iteration}.gif"
            if (folder / im_name).exists():
                msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" width=\"400\"/></a> </td>\n'.format(
                im_path_rel,
                im_path_rel,
                im_path_rel)
                f.write(msg)

            # warping rendering
            im_name = f"warping_rendering_{iteration}.gif"
            im_path_rel = f"{folder_path.name}/warping_rendering_{iteration}.gif"
            if (folder / im_name).exists():
                msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" width=\"400\"/></a> </td>\n'.format(
                im_path_rel,
                im_path_rel,
                im_path_rel)
                f.write(msg)


            f.write("\t<tr>\n")
            f.write("\t\t<td><h1>ITERATION {}</h1></td>\n".format(iteration))
            # html visu
            title = f"surface_{iteration}.html"
            f.write("\t\t<td><a href=\"./{}\">{}</a></td>\n".format(folder.name + "/" + title, title))

            # download ply
            title = f"surface_{iteration}.ply"
            f.write("\t\t<td><a href=\"./{}\" download>{}</a></td>\n".format(folder.name + "/" + title, title))

            f.write("\t</tr>\n")
            f.write("\t<tr style=\"border-top:5px solid black; padding:5px\">\n")
            f.write("\t\t<td height=20px colspan=\"100%\"></td>\n")
            f.write("\t</tr>\n")

        f.write("\t</table>\n")
        f.write("</body>\n")

