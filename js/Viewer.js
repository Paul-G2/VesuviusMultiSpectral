/**
 * @classdesc
 * This class encapsulates a pair of 2D viewports.
 * 
 */
class Viewer
{ 
    static next_instance_id = 0;

    /**
     * @constructor
     * 
     */
    constructor(site) 
    {
        this.site = site;
        this.id = Viewer.next_instance_id++;
        this.vpIndex = 0;

        // Lay out the ui
        const idPrefix = 'viewer' + this.id.toString();
        this.masterDiv = BigLime.Ui.CreateElement('div', idPrefix + '_masterDiv', site, {width:'100%', height:'100%'});
        this.imgSelectorDiv = BigLime.Ui.CreateElement('div', idPrefix + '_imgsel_div', this.masterDiv, {top:'0px', left:'0%', width:'100%', height:'50px', 
            backgroundColor:BigLime.VpTool.ActiveToolColor});
        this.vpDiv = BigLime.Ui.CreateElement('div', idPrefix + '_vp_div', this.masterDiv, {top:'50px', left:'0%', width:'100%', height:'calc(100% - 100px)'});
        this.vpToolsDiv = BigLime.Ui.CreateElement('div', idPrefix + '_vptools_div', this.masterDiv, {bottom:'0px', left:'0%', width:'100%', height:'50px'});
        this.rotAngle = this.site.dataset.rotation ? parseInt(this.site.dataset.rotation) : 0;

        const btn_pos = [5, 25, 50, 75];
        for (let i=0; i<3; i++) {
            const btn = BigLime.Ui.CreateElement('input','btn'+i.toString(), this.imgSelectorDiv, {top:20, left:btn_pos[i].toString()+'%'}, 
                {type:'radio', name: idPrefix + '_imgsel', value:i.toString(), checked:i==0});
            const btnLabel = BigLime.Ui.CreateElement('label', '', this.imgSelectorDiv, {top:14, left:(btn_pos[i]+2).toString()+'%'},  
                {innerHTML:site.dataset['label' + i.toString()]}); 
            btn.addEventListener('change', this.onRadioBtnChange.bind(this)); 
            btnLabel.addEventListener('click', function() { btn.click(); btn.focus(); } );
        }

        this.vps = []; 
        for (let i=0; i<3; i++) {
            this.vps.push( new BigLime.Viewport(this.vpDiv, {bgColor:'#000000', allowDragDrop:false}) );
            this.vps[i].displayCanvas.style.display = (i == 0) ? 'inline' : 'none';
        }
        const toolList = [new BigLime.PanTool(), new BigLime.ZoomTool(), new BigLime.WinLevelTool('Contrast'), new BigLime.PinchTool()];
        BigLime.ToolBar.BackgroundColor = "#d0d0d0";
        this.toolbar = new BigLime.ToolBar( $(this.vpToolsDiv), toolList, this.vps, 80);
        this.toolbar.getTool('zoom').wheelEnabled = 'false';
    }


    /**
     * Loads images into the Viewer.
     * 
     */
    loadImages(callback, imgIndx=2) 
    {
        const imgName = this.site.dataset['img' + imgIndx.toString()];
        if (imgName) {
            BigLime.BitmapReader.loadImage(
                'images/' + imgName, 
                function(loadResult) {
                    if (loadResult.errors) {
                        console.log(loadResult.errors);
                    }
                    this.vps[imgIndx].suspend();
                    this.vps[imgIndx].setImage(loadResult.img);
                    this.vps[imgIndx].setRotation(this.rotAngle);
                    this.vps[imgIndx].resume();
                    console.log('Loaded image ' + this.id.toString() + '_' + imgIndx.toString());
                    this.loadImages(callback, imgIndx-1);
                }.bind(this)
            );
        }
        else {
            callback();
        }
    }


    /** 
     * Handler for radio button changes
     * 
     */
    onRadioBtnChange(e) 
    {    
        const prevVp = this.vpIndex;
        const selVp = parseInt(e.target.value);

        if ((prevVp == 0) || (prevVp == 1)) {
            this.wwl01 = this.vps[prevVp].getWidthAndLevel();
        }

        this.vps[selVp].suspend();
        this.vps[selVp].setPan( this.vps[prevVp].imgLook.pan );
        this.vps[selVp].setZoom( this.vps[prevVp].imgLook.zoom );
        this.vps[selVp].setRotation( this.vps[prevVp].imgLook.rotAngle );
        if ((selVp == 0) || (selVp == 1)) {
            if (this.wwl01) {
                this.vps[selVp].setWidthAndLevel(this.wwl01.width, this.wwl01.level);
            }
        }
        this.vps[selVp].resume();

        for (let i=0; i<3; i++) {
            this.vps[i].displayCanvas.style.display = (i == selVp) ? 'inline' : 'none';
        }
        this.vpIndex = selVp;
    }
}
